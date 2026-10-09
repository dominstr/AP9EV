import numpy as np
import matplotlib.pyplot as plt

dimensions = [10, 30, 100]
runs = 10
population_size = 100
elitism_percentage = 0.14
mutation_probability = 0.008


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
        new_population = []
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

        current_best = np.max(fitness_values)
        best_fitness_history.extend([current_best] * population_size)

    best_fitness_history = best_fitness_history[:evaluations]
    best_fitness = np.max(fitness_values)

    return best_fitness, best_fitness_history


if __name__ == '__main__':
    problems = [("OneMax", fitness_one_max),
                ("LeadingOnes", fitness_leading_ones)]

    for problem_name, fitness_function in problems:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        title_text = (
            f"{problem_name} - Avg convergence over {runs} runs\n"
            f"Population size = {population_size}  Elitism percentage = {elitism_percentage * 100:.1f}%  Mutation probability = {mutation_probability * 100:.1f}%"
        )
        fig.suptitle(title_text, fontsize=12, fontweight='bold', y=0.98)

        for i, D in enumerate(dimensions):
            evaluations = 100 * D
            all_runs_final = []
            all_runs_history = []

            print(f"\n{problem_name} D={D}")
            for run in range(runs):
                # print(f"Run: {run + 1}/{runs}")
                best_value, history = genetic_algorithm(D, evaluations, fitness_function)
                all_runs_final.append(best_value)
                all_runs_history.append(history)

            mean_history = np.mean(all_runs_history, axis=0)

            ax = axes[i]
            ax.plot(mean_history, color='blue', label='Average fitness')
            ax.axhline(y=D, color='red', linestyle='--', label='Optimal fitness')
            ax.set_title(f"Dimension: {D}")
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Fitness")
            ax.grid(True)
            ax.legend()

            stats_text = (
                f"Max: {np.max(all_runs_final)}\n"
                f"Min: {np.min(all_runs_final)}\n"
                f"Mean: {np.mean(all_runs_final):.2f}\n"
                f"Median: {np.median(all_runs_final):.2f}\n"
                f"Std: {np.std(all_runs_final):.2f}"
            )

            ax.text(
                0.5, -0.32, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='whitesmoke', edgecolor='gray', alpha=0.8)
            )

            print(f"Max: {np.max(all_runs_final)}")
            print(f"Min: {np.min(all_runs_final)}")
            print(f"Mean: {np.mean(all_runs_final)}")
            print(f"Median: {np.median(all_runs_final)}")
            print(f"Std: {np.std(all_runs_final)}")

        plt.tight_layout()
        plt.show()
