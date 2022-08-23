from tracemalloc import start
from .ga import kp_ga
from .tabu import kp_tabu
import numpy as np
from multiprocessing import Pool
from timeit import default_timer as timer

def kp_hrh(weights, values, capacity, k_ga, k_tabu, l, alpha, pop_size, tournament_size, mutation_rate, tabu_pool_size, tabu_size):
    start_hrh = timer()
    sol_ga = kp_ga(weights, values, capacity, k_ga, l, alpha, pop_size, tournament_size, mutation_rate)
    end_ga = timer()

    best = 0

    with Pool(tabu_pool_size) as pool:
        res = pool.starmap(kp_tabu, [(weights, values, capacity, k_tabu, tabu_size, (value, np.array(sol))) for value, sol in sol_ga])
        best = max(res, key=lambda tup: tup[0])
    end_hrh = timer()

    print(f"\t\t\t\t\t\"ga_sol\": {max(sol_ga, key=lambda tup: tup[0])[0]},")
    print(f"\t\t\t\t\t\"ga_time\": {end_ga - start_hrh},")
    print(f"\t\t\t\t\t\"hrh_sol\": {best[0]},")
    print(f"\t\t\t\t\t\"hrh_time\": {end_hrh - start_hrh},")


    return best