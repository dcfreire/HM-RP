import random

import numpy as np

from .greedy import kp_random_greedy

from multiprocessing import Pool


def kp_ga(weights: np.ndarray, values: np.ndarray, capacity: int, k: int, l: int, alpha: float, pop_size: int, tournament_size: int, mutation_rate: float) -> set[tuple[int, tuple]]:
    n = len(weights)

    def tournament(pop: list[tuple[int, np.ndarray]]) -> tuple[tuple[int, np.ndarray], tuple[int, np.ndarray]]:
        tourn = random.choices(pop, k=tournament_size)
        p1, p2 = (-np.inf, np.empty(0)), (-np.inf, np.empty(0))
        for value, sol in tourn:
            if value > p1[0]:
                p2 = p1
                p1 = (value, sol)
            elif value > p2[0]:
                p2 = (value, sol)

        return p1, p2

    def mutate_pop(pop):
        mutants = random.sample(range(pop_size), int(mutation_rate*pop_size))
        for mutant in mutants:
            chromosome = random.randint(0, n-1)
            pop[mutant][chromosome] = int(not pop[mutant][chromosome])

    def crossover(p1, p2):
        left = random.randint(0, n-1)
        right = random.randint(left+1, n)
        s1 = np.append(p1[:left],  np.append(p2[left:right], p1[right:]))
        s2 = np.append(p2[:left],  np.append(p1[left:right], p2[right:]))
        assert len(s2) == n
        return s1, s2



    def eval_fitness(sol) -> int:
        total_weight = np.sum(weights[sol == 1])
        total_value = np.sum(values[sol == 1])
        return total_value if total_weight <= capacity else -1

    def eval_fitness_pop(pop) -> list[int]:
        return [eval_fitness(sol) for sol in pop]

    with Pool(16) as pool:
        p = [pool.starmap(kp_random_greedy, [(weights, values, capacity, alpha) for _ in range(0, pop_size)])]

    p[0].sort(key=lambda tup: tup[0], reverse=True)
    #p = [sorted([kp_random_greedy(weights, values, capacity, alpha) for _ in range(0, pop_size)], key=lambda tup: tup[0], reverse=True)]
    #p = [sorted([kp_random(weights, values, capacity) for _ in range(0, pop_size)], key=lambda tup: tup[0], reverse=True)]

    best = p[0][0][0]
    improvement = 0
    generation = 0
    while improvement < k:
        p.append([])
        fitness = []
        offspring = []
        while len(offspring) < pop_size:
            p1, p2 = tournament(p[generation])
            s1, s2 = crossover(p1[1], p2[1])
            offspring.extend([s1, s2])

        mutate_pop(offspring)
        fitness = eval_fitness_pop(offspring)
        offspring = list(zip(fitness, offspring))

        while len(p[generation+1]) < pop_size:
            combined_pop = p[generation] + offspring
            p[generation+1].append(tournament(combined_pop)[0])
        p[generation+1].sort(key=lambda tup: tup[0], reverse=True)

        improvement += 1
        if p[generation+1][0][0] > best:
            best = p[generation+1][0][0]
            improvement = 0
        generation += 1

    sol: set[tuple[int, tuple]] = set()
    ind = 0
    gen = len(p) - 1
    while len(sol) < l and gen >= 0:
        sol.add((p[gen][ind][0], tuple(p[gen][ind][1])))
        ind += 1
        if ind == len(p[0]):
            gen -= 1
            ind = 0

    return sol
