from methods import kp_ga, kp_dp, kp_tabu, kp_random_greedy, kp_hrh
import os
import numpy as np
import re
from timeit import default_timer as timer
import multiprocessing
from multiprocessing import Pool

def read_file(filepath):
    with open(filepath, 'r') as f:
        instances = []
        while line := f.readline():
            name = line[:-1]
            line = f.readline()
            n = int(line[2:])
            line = f.readline()
            c = int(line[2:])

            line = f.readline()
            z = int(line[2:])
            f.readline()

            weights = np.zeros(shape=(n,), dtype=np.int32)
            costs = np.zeros(shape=(n,), dtype=np.int32)
            for i in range(0, n):
                line = f.readline()
                _, p, w,_ = re.findall(r"\d+", line)
                weights[i] = int(w)
                costs[i] = int(p)
            instances.append((c, weights, costs, z, name))
            f.readline()
            f.readline()

        return instances


def random_start_tabu(weights: np.ndarray, values: np.ndarray, capacity: int, k: int, tabu_size: int, alpha, pop_size):
    with Pool(16) as pool:
        rgreedys = timer()
        p = pool.starmap(kp_random_greedy, [(weights, values, capacity, alpha) for _ in range(0, pop_size)])
        rgreedye = timer()
        print(f"\t\t\t\t\t\"random_greedy_time\": {rgreedye - rgreedys},")
        res = pool.starmap(kp_tabu, [(weights, values, capacity, k, tabu_size, sol) for sol in p])
        best = max(res, key=lambda tup: tup[0])
    return best, p


"""if __name__ == "__main__":
    print("{")
    for folder_name in os.listdir("test_files"):
        print(f"\t\"{folder_name}\": [")
        for file_name in os.listdir(os.path.join("test_files", folder_name)):
            if file_name == "README.txt":
                continue
            print(f"\t\t{{\n\t\t\t\"name\": \"{file_name}\",")
            instance_type, nitems, range_items = re.findall(r"(?<=\_)\d+", file_name)
            print(f"\t\t\t\"type\": {instance_type},")
            print(f"\t\t\t\"range\": {range_items},")
            dp_timeout = False

            file_path = os.path.join("test_files", folder_name, file_name)
            instances = read_file(file_path)

            ninstances = 0
            match int(nitems):
                case 10000:
                    ninstances = 5
                case 5000:
                    ninstances = 10
                case 2000:
                    ninstances = 20
                case 1000:
                    ninstances = 40
                case 500:
                    ninstances = 80
                case _:
                    ninstances = 100
            print(f"\t\t\t\"instances\": [")
            for i in range(ninstances):
                instance = instances[i * 100//ninstances]
                print("\t\t\t\t{")
                capacity, weights, values, z, name = instance
                print(f"\t\t\t\t\t\"name\": \"{name}\",")
                print(f"\t\t\t\t\t\"opt\": {z},")
                print(f"\t\t\t\t\t\"n\": {len(weights)},")
                print(f"\t\t\t\t\t\"c\": {capacity},")

                k_ga = 10
                k_tabu = 70
                l = 50
                alpha = 0.9
                pop_size = 100
                tournament_size = 5
                mutation_rate = 0.5
                tabu_size = 50

                kp_hrh(weights, values, capacity, k_ga, k_tabu, l, alpha, pop_size, tournament_size, mutation_rate, 16, tabu_size)

                start_greedy = timer()
                greedy_sol = kp_random_greedy(weights, values, capacity, 0)
                end_greedy = timer()

                print(f"\t\t\t\t\t\"greedy_sol\": {greedy_sol[0]},")
                print(f"\t\t\t\t\t\"greedy_time\": {end_greedy - start_greedy},")

                start_tabu = timer()
                tabu_sol = kp_tabu(weights, values, capacity, k_tabu, tabu_size, greedy_sol)
                end_tabu = timer()

                print(f"\t\t\t\t\t\"tabu_sol\": {tabu_sol[0]},")
                print(f"\t\t\t\t\t\"tabu_time\": {end_tabu - start_tabu},")

                start_random_tabu = timer()
                random_tabu_sol, p = random_start_tabu(weights, values, capacity, k_tabu, tabu_size, alpha, l)
                end_random_tabu = timer()

                print(f"\t\t\t\t\t\"random_greedy_sol\": {max(p, key=lambda tup: tup[0])[0]},")

                print(f"\t\t\t\t\t\"random_tabu_sol\": {random_tabu_sol[0]},")
                print(f"\t\t\t\t\t\"random_tabu_time\": {end_random_tabu - start_random_tabu},")


                if dp_timeout:
                    start_dp = timer()
                    p = multiprocessing.Process(target=kp_dp, args=(weights, values, capacity))
                    p.start()
                    p.join(1800)
                    if p.is_alive():
                        dp_timeout = True
                        print("\t\t\t\t\t\"dp_time\": -1,")
                        p.kill()
                        p.join()
                    else:
                        end_dp = timer()
                        print(f"\t\t\t\t\t\"dp_time\": {end_dp - start_dp},")
                else:
                    print("\t\t\t\t\t\"dp_time\": -1,")
                print("\t\t\t\t},")

            print("\t\t\t],")
            print("\t\t},")

        print("],")

    print("}")"""

if __name__ == "__main__":
    print("{")
    for folder_name in os.listdir("test_files"):
        print(f"\t\"{folder_name}\": [")
        for file_name in os.listdir(os.path.join("test_files", folder_name)):
            if file_name != "knapPI_1_100_10000.csv":
                continue
            if file_name == "README.txt":
                continue
            print(f"\t\t{{\n\t\t\t\"name\": \"{file_name}\",")
            instance_type, nitems, range_items = re.findall(r"(?<=\_)\d+", file_name)
            print(f"\t\t\t\"type\": {instance_type},")
            print(f"\t\t\t\"range\": {range_items},")
            dp_timeout = False

            file_path = os.path.join("test_files", folder_name, file_name)
            instances = read_file(file_path)

            ninstances = 100

            print(f"\t\t\t\"instances\": [")
            for i in range(ninstances):
                instance = instances[i * 100//ninstances]
                print("\t\t\t\t{")
                capacity, weights, values, z, name = instance
                print(f"\t\t\t\t\t\"name\": \"{name}\",")
                print(f"\t\t\t\t\t\"opt\": {z},")
                print(f"\t\t\t\t\t\"n\": {len(weights)},")
                print(f"\t\t\t\t\t\"c\": {capacity},")

                if not dp_timeout:
                    start_dp = timer()
                    p = multiprocessing.Process(target=kp_dp, args=(weights, values, capacity))
                    p.start()
                    p.join(1800)
                    if p.is_alive():
                        dp_timeout = True
                        print("\t\t\t\t\t\"dp_time\": -1,")
                        p.kill()
                        p.join()
                    else:
                        end_dp = timer()
                        print(f"\t\t\t\t\t\"dp_time\": {end_dp - start_dp},")
                else:
                    print("\t\t\t\t\t\"dp_time\": -1,")
                print("\t\t\t\t},")

            print("\t\t\t],")
            print("\t\t},")

        print("],")

    print("}")
