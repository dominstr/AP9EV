import numpy as np
import matplotlib.pyplot as plt

dimension = 10
population_size = 50
elitism_percentage = 0.15
mutation_probability = 0.05

fitness_values = []
new_population = []



def fitness_one_max(individual):
    return np.sum(individual)

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
    for bit in individual:
        if np.random.rand() < mutation_probability:
            individual[bit] = 1 - bit

    return individual


if __name__ == '__main__':
    # Create population
    population = np.random.randint(0, 2, size=(population_size, dimension))

    # Fitness evaluation
    for individual in population:
        fitness_values.append(fitness_one_max(individual))

    fitness_values = np.array(fitness_values)

    # Sort by fitness
    sorted_indices = np.argsort(fitness_values)[::-1]
    sorted_fitness = fitness_values[sorted_indices]
    sorted_population = population[sorted_indices]

    # Elitism selection
    elites_count = int(population_size * elitism_percentage)
    elites = sorted_population[:elites_count].copy()
    for individual in elites:
        individual.copy()


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
    print(population)

