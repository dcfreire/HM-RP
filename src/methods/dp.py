import numpy as np


def kp_dp(weights, values, capacity):

    n = len(weights)
    memo = np.zeros(shape=(n, capacity+1))
    for i in range(0, n):
        for j in range(0, capacity+1):
            if weights[i] > j:
                memo[i, j] = memo[i-1, j]
            else:
                memo[i, j] = max((memo[i-1, j], memo[i-1, j - weights[i]] + values[i]))

    def get_items(i, j):
        if i == 0:
            return []
        if memo[i, j] > memo[i-1, j]:
            items = get_items(i-1, j - weights[i])
            print(i+1)
            return [i].extend(items if items is not None else [])
        return get_items(i-1, j)

    return int(memo[-1, -1])
