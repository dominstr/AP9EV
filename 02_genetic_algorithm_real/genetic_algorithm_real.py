import numpy as np
import struct
import matplotlib.pyplot as plt

dimension = 10
population_size = 50
max_evaluations = 10000
elitism_percentage = 0.1
runs = 10

def schwefel_function(x):
    dim = len(x)
    return 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def bits_to_ieee754(bits_32, bounds_min, bounds_max):
    byte_str = np.packbits(bits_32).tobytes()
    try:
        value = struct.unpack("!f", byte_str)[0]
        if np.isnan(value) or np.isinf(value):
            return np.random.uniform(bounds_min, bounds_max)
        return float(value)
    except:
        return np.random.uniform(bounds_min, bounds_max)


def bits_to_fixed_point(bits_32):
    byte_str = np.packbits(bits_32).tobytes()
    signed_int = struct.unpack("!i", byte_str)[0]

    value = signed_int / 65536.0

    return float(value)

def bits_to_bcd(bits_32):
    sign_value = np.packbits(bits_32[:4])[0] >> 4
    is_negative = (sign_value >= 8)

    digits = []

    for i in range(1, 8):
        nibble = bits_32[i * 4 : (i + 1) * 4]
        digit_value = np.packbits(nibble)[0] >> 4
        if digit_value > 9:
            digit_value = 9
        digits.append(str(digit_value))

    num_str = "".join(digits[:4]) + "." + "".join(digits[4:])
    value = float(num_str)

    if is_negative:
        return -value
    else:
        return value

def decode_chromosome(chromosome_320, mode, bounds_min, bounds_max):
    decoded_vector = []

    for D in range(dimension):
        bits_32 = chromosome_320[D * 32 : (D + 1) * 32]

        if mode == "IEEE754":
            value = bits_to_ieee754(bits_32, bounds_min, bounds_max)
        elif mode == "Fixed_point":
            value = bits_to_fixed_point(bits_32)
        else:
            value = bits_to_bcd(bits_32)

        if value < bounds_min:
            value = bounds_min
        if value > bounds_max:
            value = bounds_max

        decoded_vector.append(value)

    return np.array(decoded_vector)

def rank_selection(sorted_population):
    ranks = np.arange(population_size, 0, -1)
    probabilities = ranks / np.sum(ranks)
    random_indices = np.random.choice(population_size, 2, p=probabilities)

    return sorted_population[random_indices[0]].copy(), sorted_population[random_indices[1]].copy()


def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2

def mutate_binary(individual):
    mutated = individual.copy()
    p_mutation = 1 / len(mutated)

    for i in range(len(mutated)):
        if np.random.rand() <p_mutation:
            mutated[i] = 1 -mutated[i]
    return mutated

def mutate_real_gaussian(individual, bounds_min, bounds_max, p_mutation=0.1, sigma=0.5):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < p_mutation:
            mutated[i] += np.random.normal(0, sigma)
            if mutated[i] > bounds_max:
                mutated[i] = bounds_max
            if mutated[i] < bounds_min:
                mutated[i] = bounds_min
    return mutated

def mutate_real_uniform(individual, bounds_min, bounds_max, p_mutation=0.1):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < p_mutation:
            mutated[i] = np.random.uniform(bounds_min, bounds_max)

    return mutated


def genetic_algorithm(mode, evaluations, fitness_function, bounds_min, bounds_max):
    if mode in ["IEEE754", "Fixed_point", "BCD"]:
        is_binary = True
    else:
        is_binary = False

    fitness_values = []

    if is_binary:
        population = np.random.randint(0, 2, size=(population_size, dimension * 32))

        for index in population:
            fitness_values.append(fitness_function(decode_chromosome(index, mode, bounds_min, bounds_max)))

    else:
        population = np.random.uniform(bounds_min, bounds_max, size=(population_size, dimension))
        for index in population:
            fitness_values.append(fitness_function(index))

    evaluations_count = population_size
    best_fitness_history = []

    current_best = np.min(fitness_values)
    best_fitness_history.extend([current_best] * population_size)
    elites_count = int(population_size * elitism_percentage)

    while evaluations_count < evaluations:
        sorted_indices = np.argsort(fitness_values)
        fitness_values = np.array(fitness_values)
        sorted_fitness = fitness_values[sorted_indices]
        sorted_population = population[sorted_indices]

        # Elitism
        new_population = [ind.copy() for ind in sorted_population[:elites_count]]

        # Doplnění populace
        while len(new_population) < population_size:
            parent1, parent2 = rank_selection(sorted_population)
            child1, child2 = single_point_crossover(parent1, parent2)

            if is_binary:
                child1 = mutate_binary(child1)
                child2 = mutate_binary(child2)
            else:
                if mode == "Gaussian":
                    child1 = mutate_real_gaussian(child1, bounds_min, bounds_max)
                    child2 = mutate_real_gaussian(child2, bounds_min, bounds_max)
                else:
                    child1 = mutate_real_uniform(child1, bounds_min, bounds_max)
                    child2 = mutate_real_uniform(child2, bounds_min, bounds_max)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)

        new_fitness = list(sorted_fitness[:elites_count])
        for i in range(elites_count, population_size):
            if is_binary:
                real_x = decode_chromosome(population[i], mode, bounds_min, bounds_max)
                fit = fitness_function(real_x)
            else:
                fit = fitness_function(population[i])

            new_fitness.append(fit)
            evaluations_count += 1
            best_fitness_history.append(np.min(new_fitness))

            if evaluations_count >= evaluations:
                break

        fitness_values = np.array(new_fitness)

    best_fitness_history = best_fitness_history[:evaluations]
    best_fitness = np.min(fitness_values)

    return best_fitness, best_fitness_history


if __name__ == "__main__":
    functions = [
        ("Schwefell", schwefel_function, -500, 500),
        ("Rastrigin", rastrigin_function, -5.12, 5.12),
        ("Rosenbrock", rosenbrock_function, -2.048, 2.048),
    ]

    modes = ["IEEE754", "Fixed_point", "BCD", "Gaussian", "Uniform"]

    for function, fitness_function, b_min, b_max in functions:
        plt.figure(figsize=(9,5))

        for mode in modes:
            all_runs_final = []
            all_runs_history = []

            for run in range(runs):
                best_value, history = genetic_algorithm(mode, max_evaluations, fitness_function, b_min, b_max)
                all_runs_final.append(best_value)
                all_runs_history.append(history)

            mean_history = np.mean(all_runs_history, axis=0)
            plt.plot(mean_history, label=mode)
            print(
                f"Variant: {mode:12s} | Min: {np.min(all_runs_final):.4f} | Max: {np.max(all_runs_final):.4f} | Mean: {np.mean(all_runs_final):.4f} | Median: {np.median(all_runs_final):.4f} | Std: {np.std(all_runs_final):.4f}")

        plt.yscale('log')
        plt.title(f"Convergence - {function} (10D)")
        plt.xlabel("Evaluations")
        plt.ylabel("Fitness (Log)")
        plt.grid(True)
        plt.legend()
        plt.show()

