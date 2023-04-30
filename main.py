import random
import numpy as np
import CoolProp.CoolProp as coolprop

# Define the objective function to be optimized


def objective_function(x):
    T4, Rh4 = x
    T4 += 273.15
    T2, RH2 = point2(T4, Rh4)
    T2_GOAL, RH2_GOAL = 25 + 273.15, 0.55
    return abs(T2 - T2_GOAL)/T2_GOAL*100 + abs(RH2 - RH2_GOAL)/1*100


# Define the bounds for the decision variables
bounds = ((10, 25), (0, 1))

# Define the size of the population and the number of generations
pop_size = 100
num_generations = 1000

# Define the mutation rate and the crossover rate
mutation_rate = 0.1
crossover_rate = 0.8

# Define the function to generate an initial population


def generate_population(bounds, pop_size):
    population = []
    for i in range(pop_size):
        individual = []
        for j in range(len(bounds)):
            individual.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(individual)
    return population

# Define the function to evaluate the fitness of each individual in the population


def evaluate_fitness(population):
    fitness = []
    for individual in population:
        fitness.append(objective_function(individual))
    return np.array(fitness)

# Define the function to select parents for crossover


def select_parents(population, fitness):
    # Select two parents using tournament selection
    parent_1, parent_2 = random.choices(population, weights=fitness, k=2)
    return parent_1, parent_2

# Define the function to perform crossover


def crossover(parent_1, parent_2):
    # Select a crossover point
    crossover_point = random.randint(0, len(parent_1) - 1)
    # Create a new offspring by combining the parents' genes
    offspring = parent_1[:crossover_point] + parent_2[crossover_point:]
    return offspring

# Define the function to perform mutation


def mutation(offspring, bounds, mutation_rate):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] = random.uniform(bounds[i][0], bounds[i][1])
    return offspring


def point2(T4, RH4):
    Q4 = 8.333333333
    Qsr = 100  # kW
    Qlr = 30  # kW
    Qvent = 70  # kW
    T1 = 33.2 + 273.15
    T2_goal = 25 + 273.15
    Qvent_s = 1.1 * 6000 * (T1 - T2_goal) * 9/5  # Btu/Hr
    Qvent_s = Qvent_s * 2.931 * 10 ** -4  # kW
    Qvent_l = Qvent - Qvent_s  # kW
    Qst = Qsr + Qvent_s  # kW
    Qlt = Qlr + Qvent_l  # kW

    v4 = coolprop.HAPropsSI('V', 'T', T4, 'P', 101325, 'R', RH4)
    m4_dot = Q4/v4

    T2 = (Qst/m4_dot) + T4

    w4 = coolprop.HAPropsSI('W', 'T', T4, 'P', 101325, 'R', RH4)
    hlr = coolprop.HAPropsSI('H', 'T', T2, 'P', 101325, 'W', w4)/1000
    h2 = (Qlt/m4_dot) + hlr
    RH2 = coolprop.HAPropsSI('R', 'T', T2, 'P', 101325, 'H', h2*1000)

    return T2, RH2

    # Generate an initial population
population = generate_population(bounds, pop_size)

# Iterate through the generations
best = []
best_fitness = 100

for i in range(num_generations):
    # Evaluate the fitness of the population
    fitness = evaluate_fitness(population)
    # Select the best individual from the population
    best_individual = population[np.argmin(fitness)]
    # Print the best individual and its fitness
    print(
        f"Generation {i}: Best Individual = {best_individual}, Fitness = {min(fitness)}")
    if best_fitness > min(fitness):
        best = best_individual
        best_fitness = min(fitness)
    # Create a new population
    new_population = []
    # Perform selection, crossover, and mutation to create the new population
    while len(new_population) < pop_size:
        # Select two parents for crossover
        parent_1, parent_2 = select_parents(population, fitness)
        # Perform crossover to create an offspring
        offspring = crossover(parent_1, parent_2)
        # Perform mutation on the offspring
        offspring = mutation(offspring, bounds, mutation_rate)
        # Add the offspring to the new population
        new_population.append(offspring)
    # Update the population
    population = new_population

print(best, best_fitness)
print(point2(best[0] + 273.15, best[1]))
