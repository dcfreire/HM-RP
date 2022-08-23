import numpy as np

def kp_tabu(weights: np.ndarray, values: np.ndarray, capacity: int, k: int, tabu_size: int, initial_solution: tuple[int, np.ndarray]):

    def get_neighbor(sol, tabu_list, iteration):
        best = (0, None)
        best_weight = 0
        sol_value = sol[0]
        sol_weight = np.sum(weights[sol[1]==1])
        idx = -1
        for i in range(len(sol[1])):

            cur = sol[1].copy()
            cur[i] = int(cur[i] != 1)

            cur_value = sol_value + values[i]*cur[i] - values[i]*int(cur[i] == 0)
            cur_weight = sol_weight + weights[i]*cur[i] - weights[i]*int(cur[i] == 0)

            if cur_weight > capacity:
                cur_value = -1

            if tabu_list.get((cur_value, cur_weight), 0) > iteration:
                continue

            if cur_value > best[0]:
                best = (cur_value, cur)
                best_weight = cur_weight
                idx = i

        if idx >= 0 :
            tabu_list[(best[0], best_weight)] = iteration + tabu_size

        return best if best[1] is not None else sol

    tabu_list = {}


    prev = initial_solution
    best = initial_solution
    improvement = 0
    iteration = 0
    while iteration - improvement < k:
        iteration += 1
        value, sol = get_neighbor(prev, tabu_list, iteration)
        prev = (value, sol)
        if value > best[0]:
            improvement = iteration
            best = prev

    return best
