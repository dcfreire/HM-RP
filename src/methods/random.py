import numpy as np
import random

def kp_random(weights: np.ndarray, values: np.ndarray, capacity: int):
    ret = np.zeros((weights.size,))
    inds = list(range(weights.size))
    random.shuffle(inds)
    for item in inds:
        total = np.sum(weights[ret == 1])
        if total + weights[item] < capacity:
            ret[item] = 1
        elif total >= capacity:
            break
    return np.sum(values[ret == 1]), ret
