"""Group-relative interval localization; no classification threshold."""
from math import isclose

from .attribution import shapley_values


def interval_mask(left, right):
    return ((1 << (right - left)) - 1) << left


def relative_split(entries, focus="high"):
    """Split at the unique largest adjacent gap; no absolute probe threshold."""
    sign = 1 if focus == 'high' else -1
    ordered = sorted(entries, key=lambda row: (sign * row['phi'], row['index']))
    if len(ordered) < 2:
        return {'status': 'insufficient_comparison', 'selected_mask': 0, 'largest_gap': None,
                'runner_up_gap': None, 'gap_ratio': None}
    gaps = [sign * (b['phi'] - a['phi']) for a, b in zip(ordered, ordered[1:])]
    largest = max(gaps)
    # Only relative tolerance: an absolute epsilon would change decisions when
    # the same effects are expressed at a different positive scale.
    positions = [i for i, gap in enumerate(gaps)
                 if isclose(gap, largest, rel_tol=1e-9, abs_tol=0.0)]
    if largest == 0 or len(positions) != 1:
        return {'status': 'no_unique_separation', 'selected_mask': 0, 'largest_gap': largest,
                'runner_up_gap': None, 'gap_ratio': None}
    cut = positions[0]
    other = max((gap for i, gap in enumerate(gaps) if i != cut), default=0)
    return {'status': 'separated',
            'selected_mask': sum(1 << (row['index'] - 1) for row in ordered[cut + 1:]),
            'largest_gap': largest, 'runner_up_gap': other,
            'gap_ratio': largest / other if other > 0 else None,
            'sorted_indices': [row['index'] for row in ordered]}


def group_effects(n, group, score_mask, focus="high"):
    """Conditional Shapley within this group; all outside paragraphs stay present."""
    left, right = group
    players = list(range(left, right))
    outside = ((1 << n) - 1) ^ interval_mask(left, right)
    values = {}
    for local in range(1 << len(players)):
        mask = outside | sum(1 << p for i, p in enumerate(players) if local & (1 << i))
        values[local] = score_mask(mask)
    phi = shapley_values(values, len(players))
    low, high = min(phi), max(phi)
    entries = [{'index': p + 1, 'phi': value,
                'relative': (value - low) / (high - low) if high != low else None}
               for p, value in zip(players, phi, strict=True)]
    return {'group': group, 'outside_mask': outside, 'minimum': low, 'maximum': high,
            'values': entries, 'split': relative_split(entries, focus),
            'definition': '(conditional Shapley - group minimum) / (group maximum - group minimum)'}


def locate_interval(n, score_mask, focus="high"):
    """Seed from group outliers, expand boundary controls, then shrink non-seed edges.

    Decisions use group ranks and gaps only. Ambiguous groups remain ambiguous;
    an unstable conditional ranking is reported rather than silently discarded.
    """
    full = (1 << n) - 1
    cache, queries, frames, groups = {}, [], [], {}

    def score(mask):
        if mask not in cache:
            cache[mask] = float(score_mask(mask))
            queries.append(mask)
        return cache[mask]

    def group(bounds):
        key = tuple(bounds)
        if key not in groups:
            groups[key] = group_effects(n, bounds, score, focus)
        return groups[key]

    def record(phase, title, window, bounds, note, stable=None):
        frames.append({'phase': phase, 'title': title, 'window': window,
                       'heat': group(bounds) if bounds is not None else None,
                       'note': note, 'stable': stable})

    record('baseline', '准备组内相对分析', None, None,
           '只比较段落在组内的影响，不判断是否越过某个绝对分数。')
    root = group([0, n])
    core = root['split']['selected_mask']
    if not core:
        record('done', '本组没有唯一的离群分割', None, [0, n],
               '相同影响或并列间隔不强行选出关键段。', False)
        return {'status': root['split']['status'], 'frames': frames, 'queries': queries,
                'window': None, 'mask': 0, 'core_mask': 0, 'stable_across_groups': False}
    indices = [i for i in range(n) if core & (1 << i)]
    anchor = min(indices)
    left, right = anchor, max(indices) + 1
    record('candidate', f'候选起点：第 {anchor + 1} 段', [left, right], [0, n],
           '按最大相邻间隔识别目标离群簇；候选区间覆盖该簇，不预设只含一段。')
    if left:
        left -= 1
        record('expand', '向左扩展比较组', [left, right], [left, right],
               '纳入左邻段，重新计算条件 Shapley，观察目标离群簇是否保持。')
    if right < n:
        right += 1
        record('expand', '向右扩展比较组', [left, right], [left, right],
               '纳入右邻段，色条仅在这个比较组内归一化。')
    stable = True
    while right - left > 1:
        current = group([left, right])
        ranked = current['split']['selected_mask']
        stable = stable and ranked == core
        if not ranked:
            break
        if not (core & (1 << left)) and not (ranked & (1 << left)):
            left += 1
            record('shrink', '收缩左边界：左邻段不是组内目标离群簇', [left, right],
                   [left, right] if right - left > 1 else current['group'],
                   '移出比较区间，不把相对非目标的配合上下文圈成核心。', ranked == core)
        elif not (core & (1 << (right - 1))) and not (ranked & (1 << (right - 1))):
            right -= 1
            record('shrink', '收缩右边界：右邻段不是组内目标离群簇', [left, right],
                   [left, right] if right - left > 1 else current['group'],
                   '保留粗组目标离群簇成员，避免把跨段候选强行缩成一段。', ranked == core)
        else:
            break
    for a, b in dict.fromkeys([(max(0, left - 1), right), (left, min(n, right + 1))]):
        if b - a < 2:
            continue
        consistent = group([a, b])['split']['selected_mask'] == core
        stable = stable and consistent
        record('verify', f'相邻比较组 §{a + 1}—§{b}：验证离群簇', [left, right], [a, b],
               '组外段落固定；核对目标离群簇是否仍指向同一核心。', consistent)
    record('done', '区间定位完成，和完整枚举对照', [left, right], [0, n],
           '色条回到完整比较组；蓝色只是组内相对较低，不等于负贡献或安全。', stable)
    return {'status': 'localized' if stable else 'conditional_ranking_unstable',
            'anchor': anchor, 'window': [left, right], 'mask': interval_mask(left, right),
            'core_mask': core, 'stable_across_groups': stable,
            'frames': frames, 'queries': queries, 'root_split': root['split'],
            'objective': 'contiguous bounds around a group-relative high-influence cluster'}
