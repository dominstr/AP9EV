import numpy as np
import matplotlib.pyplot as plt

dimensions = [10, 30, 100]
runs = 10
population_size = 50
elitism_percentage = 0.15
mutation_probability = 0.05


def fitness_one_max(individual):
    return np.sum(individual)


def fitness_leading_ones(individual):
    count = 0
    for bit in individual:
        if bit == 1:
            count += 1
        else:
            break
    return count

def rank_selection(sorted_population):
    ranks = np.arange(population_size, 0, - 1)
    probabilities = ranks / np.sum(ranks)
    random_indices = np.random.choice(population_size, 2, p=probabilities)

    return sorted_population[random_indices[0]].copy(), sorted_population[random_indices[1]].copy()


def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2


def mutate(individual):
    for i, bit in enumerate(individual):
        if np.random.rand() < mutation_probability:
            individual[i] = 1 - bit

    return individual


def genetic_algorithm(dimension, evaluations, fitness_function):
    fitness_values = []
    new_population = []
    evaluations_count = population_size

    # Create population
    population = np.random.randint(0, 2, size=(population_size, dimension))

    # Fitness evaluation
    for individual in population:
        fitness_values.append(fitness_function(individual))

    fitness_values = np.array(fitness_values)

    best_fitness_history = []
    current_best = np.max(fitness_values)
    best_fitness_history.extend([current_best] * population_size)
    elites_count = int(population_size * elitism_percentage)

    while evaluations_count < evaluations:
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        sorted_fitness = fitness_values[sorted_indices]
        sorted_population = population[sorted_indices]

        # Elitism selection
        elites = sorted_population[:elites_count].copy()
        for elite in elites:
            new_population.append(elite.copy())

        while len(new_population) < population_size:
            # Crossover
            parent1, parent2 = rank_selection(sorted_population)
            child1, child2 = single_point_crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)

        fitness_values = []
        for individual in population:
            fitness_values.append(fitness_function(individual))

        fitness_values = np.array(fitness_values)

        evaluations_count += population_size

    return np.max(fitness_values)


if __name__ == '__main__':
    problems = [("OneMax", fitness_one_max), ("LeadingOnes", fitness_leading_ones)]

    for D in dimensions:
        evaluations = 100 * D
        all_runs_onemax = []

        print(f"\nOneMax D={D}")
        for run in range(runs):
            print(f"Run: {run + 1}/{runs}")
            best_values = genetic_algorithm(D, evaluations, fitness_one_max)
            all_runs_onemax.append(best_values)

        print(f"Max: {np.max(all_runs_onemax)}")
        print(f"Min: {np.min(all_runs_onemax)}")
        print(f"Mean: {np.mean(all_runs_onemax)}")
        print(f"Median: {np.median(all_runs_onemax)}")
        print(f"Std: {np.std(all_runs_onemax)}")

        print(f"\nLeadingOnes D={D}")
        all_runs_leadingones = []

        for run in range(runs):
            print(f"Run: {run + 1}/{runs}")
            best_values = genetic_algorithm(D, evaluations, fitness_leading_ones)
            all_runs_leadingones.append(best_values)

        print(f"Max: {np.max(all_runs_leadingones)}")
        print(f"Min: {np.min(all_runs_leadingones)}")
        print(f"Mean: {np.mean(all_runs_leadingones)}")
        print(f"Median: {np.median(all_runs_leadingones)}")
        print(f"Std: {np.std(all_runs_leadingones)}")
