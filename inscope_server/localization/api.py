"""Public observation-debugging API, independent of an inference server."""
from copy import deepcopy
from bisect import bisect_left
from math import isfinite

from .attribution import interaction_terms
from .localization import locate_interval
from .segments import Span, coalition_text, split_spans


async def analyze_observation(request, observation_index, score_batch, *, case_id='observation',
                              spans=None, granularity='paragraph', mode='exact', focus='high',
                              group_size=6, max_coalitions=256):
    """Exact or explicitly hierarchical group-relative counterfactual analysis.

    The async scorer accepts equally ordered request batches and returns dicts
    containing finite ``logit``. Hierarchical mode is exact within each bounded
    group, not a claim of full fine-unit Shapley or complete global localization.
    """
    if focus not in ('high', 'low') or mode not in ('exact', 'hierarchical'):
        raise ValueError('focus must be high/low and mode must be exact/hierarchical')
    if max_coalitions < 1 or group_size < 2:
        raise ValueError('positive coalition budget and group_size>=2 are required')
    snapshot = deepcopy(request)
    text = snapshot['messages'][observation_index]['content']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('observation content must be nonempty text')
    atoms = tuple(split_spans(text, granularity) if spans is None else spans)
    previous = 0
    for span in atoms:
        if not 0 <= previous <= span.start < span.end <= len(text):
            raise ValueError('spans must be ordered, nonempty and nonoverlapping Unicode intervals')
        previous = span.end
    if not atoms:
        raise ValueError('at least one observation unit is required')
    newlines = [i for i, char in enumerate(text) if char == '\n']

    def line_number(position):
        return bisect_left(newlines, position) + 1
    root_size = len(atoms) if mode == 'exact' else min(group_size, len(atoms))
    if root_size >= max_coalitions.bit_length():
        raise ValueError(f'{len(atoms)} units exceed exact coalition budget {max_coalitions}; select a smaller group, explicit hierarchical mode or a larger budget')

    cache, levels, leaves = {}, [], []
    coalition_count = 0
    score_batches = 0
    wave_number = 0
    model_prompts = 0
    model_metadata_known = True
    projected_input_collapsed = False
    # Each pending branch retains its selected coarse region if budget or a
    # finer ambiguous group prevents resolution. No branch silently disappears.
    pending = [(0, len(atoms), None, None)]
    while pending:
        wave, pending = pending, []
        plans, missing_wave = [], {}
        for left, right, parent, inherited in wave:
            count = right - left
            k = count if mode == 'exact' else min(group_size, count)
            bounds = [(left + i * count // k, left + (i + 1) * count // k) for i in range(k)]
            units = [Span(atoms[a].start, atoms[b - 1].end) for a, b in bounds]
            texts = [coalition_text(text, units, mask) for mask in range(1 << k)]
            missing = list(dict.fromkeys(content for content in texts if content not in cache and content not in missing_wave))
            if len(cache) + len(missing_wave) + len(missing) > max_coalitions:
                if parent is None:
                    raise ValueError('root group exceeds coalition budget before inference')
                leaves.append({**inherited, 'resolution': 'budget_limited'})
                continue
            missing_wave.update(dict.fromkeys(missing))
            plans.append((left, right, parent, inherited, k, bounds, units, texts))
        contents = list(missing_wave)
        if contents:
            requests = []
            for content in contents:
                messages = list(snapshot['messages'])
                messages[observation_index] = dict(messages[observation_index], content=content)
                requests.append({**snapshot, 'messages': messages})
            records = await score_batch(requests)
            if len(records) != len(requests):
                raise ValueError(f'scorer returned {len(records)} rows for {len(requests)} requests')
            if any(not isfinite(float(row['logit'])) for row in records):
                raise ValueError('scorer returned a nonfinite logit; no attribution can be computed')
            known = all('prompt_index' in row for row in records)
            model_metadata_known = model_metadata_known and known
            model_prompts += len({row['prompt_index'] for row in records}) if known else 0
            if wave_number == 0 and known and len({row['prompt_index'] for row in records}) == 1 and len(contents) > 1:
                projected_input_collapsed = True
            cache.update((content, {**row, 'score_batch': score_batches})
                         for content, row in zip(contents, records, strict=True))
            score_batches += 1
        for left, right, parent, inherited, k, bounds, units, texts in plans:
            coalition_count += 1 << k
            queries = [{**cache[content], 'mask': mask, 'logit': float(cache[content]['logit'])}
                       for mask, content in enumerate(texts)]
            values = {row['mask']: row['logit'] for row in queries}
            search = locate_interval(k, values.__getitem__, focus)
            if projected_input_collapsed and parent is None:
                search['status'] = 'identical_model_input'
                search['frames'][-1].update(title='干预未改变模型实际输入',
                    note='不同文本映射到同一token序列；请检查所选消息、上下文裁剪或归一化。')
            root = search['frames'][1]['heat']
            phi = {row['index']: row for row in root['values']}
            segments = [{'index': i + 1, 'start': s.start, 'end': s.end, 'text': text[s.start:s.end],
                         'phi': phi[i + 1]['phi'], 'relative': phi[i + 1]['relative'],
                         'line_start': line_number(s.start), 'line_end': line_number(s.end - 1),
                         'atom_start': bounds[i][0], 'atom_end': bounds[i][1]}
                        for i, s in enumerate(units)]
            level_id = len(levels)
            level = {'id': level_id, 'parent': parent, 'wave': wave_number, 'atom_range': [left, right], 'segments': segments,
                     'search': search, 'queries': queries, 'interactions': interaction_terms(values, k),
                     'empty_logit': values[0], 'full_logit': values[(1 << k) - 1],
                     'efficiency_residual': sum(s['phi'] for s in segments) - (values[(1 << k) - 1] - values[0])}
            levels.append(level)
            selected = [s for s in segments if search['core_mask'] & (1 << (s['index'] - 1))]
            if not selected and inherited is not None:
                leaves.append({**inherited, 'resolution': 'no_unique_child'})
            for segment in selected:
                region = {**segment, 'level_id': level_id}
                if mode == 'hierarchical' and segment['atom_end'] - segment['atom_start'] > 1:
                    pending.append((segment['atom_start'], segment['atom_end'], level_id, region))
                else:
                    leaves.append({**region, 'resolution': 'atomic'})
        wave_number += 1
    first = levels[0]
    unresolved = any(r['resolution'] != 'atomic' for r in leaves)
    unstable = any(level['search']['status'] == 'conditional_ranking_unstable' for level in levels)
    status = ('resolution_limited' if unresolved else 'conditional_ranking_unstable' if unstable
              else first['search']['status'])
    leaves.sort(key=lambda r: r['start'])
    interval = (dict(start=leaves[0]['start'], end=max(r['end'] for r in leaves),
                     atom_start=min(r['atom_start'] for r in leaves), atom_end=max(r['atom_end'] for r in leaves),
                     line_start=line_number(leaves[0]['start']),
                     line_end=line_number(max(r['end'] for r in leaves) - 1))
                if leaves else None)
    return {'schema': 'probe_training.debug.observation.v1', 'case_id': case_id,
            'focus': focus, 'status': status, 'observation_index': observation_index,
            'granularity': 'custom' if spans is not None else granularity, 'mode': mode,
            'offset_unit': 'Unicode codepoint; zero-based, end-exclusive', 'request': snapshot,
            'atoms': [{'index': i + 1, 'start': s.start, 'end': s.end, 'text': text[s.start:s.end]}
                      | {'line_start': line_number(s.start), 'line_end': line_number(s.end - 1)}
                      for i, s in enumerate(atoms)],
            'segments': first['segments'], 'regions': leaves, 'interval': interval, 'levels': levels,
            'search': first['search'], 'queries': first['queries'], 'interactions': first['interactions'],
            'empty_logit': first['empty_logit'], 'full_logit': first['full_logit'],
            'total_logit_change': first['full_logit'] - first['empty_logit'],
            'efficiency_residual': first['efficiency_residual'],
            'exactness': 'full selected-unit game' if mode == 'exact' else 'exact per group; hierarchical localization',
            'cost': {'coalitions': coalition_count, 'unique_inputs': len(cache), 'score_batches': score_batches,
                     'model_requests': model_prompts if model_metadata_known else None}}
