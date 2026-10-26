"""Exact small-group Shapley attribution in logit space."""
from math import comb


def subsets(mask):
    value = mask
    while True:
        yield value
        if value == 0:
            break
        value = (value - 1) & mask


def shapley_values(values, n):
    full = (1 << n) - 1
    result = []
    weights = [1 / (n * comb(n - 1, k)) for k in range(n)]
    for i in range(n):
        bit = 1 << i
        phi = 0.0
        for mask in subsets(full ^ bit):
            k = mask.bit_count()
            weight = weights[k]
            phi += weight * (values[mask | bit] - values[mask])
        result.append(phi)
    return result


def interaction_terms(values, n):
    return [{'mask': mask, 'members': [i + 1 for i in range(n) if mask & (1 << i)],
             'order': mask.bit_count(),
             'logit': sum((-1) ** (mask.bit_count() - sub.bit_count()) * values[sub]
                          for sub in subsets(mask))}
            for mask in range(1, 1 << n)]
