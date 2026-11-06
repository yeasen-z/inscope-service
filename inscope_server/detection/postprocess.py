"""Calibrated policies; no threshold fitting or GT-guided expansion at serving."""
import math


class CalibratedPostProcessor:
    def __init__(self,entries):
        self.entries = list(entries)
        names = [e['name'] for e in self.entries]
        if not names or len(set(names)) != len(names):
            raise ValueError('Unique detector names required')
        for e in self.entries:
            if not math.isfinite(e['logit_threshold']):
                raise ValueError('Finite frozen logit thresholds required')

    def process(self,scores,prompt,plan):
        result = {}
        enabled=prompt.metadata.get('enabled_detectors')
        for i,entry in enumerate(self.entries):
            if enabled is not None and entry['name'] not in enabled:continue
            threshold = entry['logit_threshold']
            value = float(scores['logits'][0,i])
            detector = scores['detector_logits']
            detector_logit = None if detector is None else float(detector[0,i])
            if prompt.metadata.get('output')=='scores':
                result[entry['name']]=dict(layer=entry['layer'],logit=value,logit_threshold=threshold,flagged=value>=threshold,
                    probability=1/(1+math.exp(-max(-700,min(700,value)))),threshold_source=entry.get('threshold_source','frozen detector calibration'),
                    region_logit=float(scores['region_logits'][0,i]),detector_logit=detector_logit)
                continue
            token_values=scores['token_logits'][:,i].tolist()
            spans=[]
            token_threshold = entry.get('token_logit_threshold', threshold)
            for p,z in zip(plan.region[1:],token_values[1:]):
                if z>=token_threshold:
                    if spans and spans[-1][1]==p:spans[-1][1]=p+1
                    else:spans.append([p,p+1])
            result[entry['name']] = dict(layer=entry['layer'],logit=value,token_spans=spans,
                probability=1/(1+math.exp(-max(-700,min(700,value)))),
                logit_threshold=threshold,flagged=value>=threshold,
                region_logit=float(scores['region_logits'][0,i]),
                detector_logit=detector_logit,
                token_logits=token_values,
                token_positions=list(plan.region),
                threshold_source=entry.get('threshold_source','frozen detector calibration'))
        return dict(composer_id=prompt.composer_id,input_tokens=len(prompt.input_ids),detectors=result)
