import numpy as np
import matplotlib.pyplot as plt

dim = 10
tau1 = tau2 = 0.1
CR = 0.9
F = 0.5
pop_size = 30
max_generations = 150


def func_sphere(x):
    return np.sum(x**2)


def rastrigin(x: np.ndarray) -> float:
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e)


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

functions = [
    ("Sphere", sphere, [(-100.0, 100.0)] * dim),
    ("Rastrigin", rastrigin, [(-5.12, 5.12)] * dim),
    ("Ackley", ackley, [(-32.768, 32.768)] * dim),
]

variants = [
        ('rand1', "Rand/1"),
        ('best1', "Best/1"),
        ('jDE',   "jDE"),
    ]


def differential_evolution(
        func,
        bounds,
        variant = "jDE"
):
    bounds_array = np.array(bounds)
    lower_b, upper_b = bounds_array[:, 0], bounds_array[:, 1]

    # random population
    pop = lower_b + np.random.rand(pop_size, dim) * (upper_b - lower_b)
    fitness_list = []
    for p in pop:
        fitness_list.append(func(p))

    fitness = np.array(fitness_list)

    # jDE autoadaptation
    jde_F = np.full(pop_size, F)
    jde_CR = np.full(pop_size, 0.9)
    history = []

    for generation in range(max_generations):
        best_idx = np.argmin(fitness)
        history.append(float(fitness[best_idx]))

        new_pop, new_fitness = np.copy(pop), np.copy(fitness)

        for i in range(pop_size):
            # parameter adaptation
            if variant == "jdE":
                if np.random.random() < tau1:
                    F_i = 0.1 + np.random.rand() * 0.8
                else:
                    F_i = jde_F[i]

                if np.random.rand() < tau2:
                    CR_i = np.random.rand()
                else:
                    CR_i = jde_CR[i]
            else:
                F_i = F
                CR_i = CR

            candidates = []
            for idx in range(pop_size):
                if idx != i:
                    candidates.append(idx)

            if variant == "best1":
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                v_i = pop[best_idx] + F_i * (pop[r1] - pop[r2])
            else:
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                v_i = pop[r1] + F_i * (pop[r2] - pop[r3])

            # crossover
            j_rand = np.random.randint(0, dim)
            cross_mask = np.random.rand(dim) < CR_i
            cross_mask[j_rand] = True
            u_i = np.where(cross_mask, v_i, pop[i])

            u_i = np.clip(u_i, lower_b, upper_b)

            # selection
            f_u = func(u_i)
            if f_u <= fitness[i]:
                new_pop[i] = u_i
                new_fitness[i] = f_u
                if variant == "jdE":
                    jde_F[i] = F_i
                    jde_CR[i] = CR_i


        pop = new_pop
        fitness = new_fitness

    best_idx = np.argmin(fitness)
    return pop[best_idx], float(fitness[best_idx]), history


if __name__ == '__main__':

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (func_name, func, bounds) in zip(axes, functions):
        for variant, variant_name in variants:
            best_solution, best_fitness, history = differential_evolution(func, bounds, variant)
            ax.plot(history, label=f"{variant_name} (Best Fitness: {best_fitness:.6f})")

        ax.set_title(f"{func_name} Function")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Best Fitness")
        ax.legend()
        ax.grid()

    plt.tight_layout()

    plt.show()


    for func_name, func, bounds in functions:
        print(f"Function: {func_name}")
        for variant, variant_name in variants:
            best_solution, best_fitness, history = differential_evolution(func, bounds, variant)
            print(f"Variant: {variant_name}, Best Fitness: {best_fitness:.6f}, Best Solution: {best_solution}")
        print()
