import random
import numpy as np

def kp_random_greedy(weights: np.ndarray, values: np.ndarray, capacity: int, alpha: float) -> tuple[int, np.ndarray]:
    cost_density = values/weights
    ret_items = np.zeros((len(values),), dtype=np.int8)
    inds = cost_density.argsort()[::-1]
    cost_density = cost_density[inds]
    weights = weights[inds]
    values = values[inds]
    sol = 0
    items = []
    while weights.size:
        lrc_cutoff = cost_density[0] - alpha*(cost_density[0] - cost_density[-1])
        cutoff_idx = cost_density[cost_density >= lrc_cutoff].size - 1
        item = random.randint(0, cutoff_idx)


        if capacity - weights[item] > 0:
            items.append(inds[item])
            sol += values[item]
            capacity -= weights[item]

        weights = np.delete(weights, item)
        values = np.delete(values, item)
        inds = np.delete(inds, item)
        cost_density = np.delete(cost_density, item)


    ret_items[items] = 1
    return sol, ret_items
