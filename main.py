import random
import numpy as np
import CoolProp.CoolProp as coolprop


def objective_function(x):
    T4, Rh4 = x
    T4 += 273.15
    T2, RH2 = point2(T4, Rh4)
    T2_GOAL, RH2_GOAL = 25 + 273.15, 0.55
    return abs(T2 - T2_GOAL)/T2_GOAL*200**2 + abs(RH2 - RH2_GOAL)/1*100**2


bounds = ((10, 100), (0, 1))
pop_size = 100
num_generations = 2000


def generate_population(bounds, pop_size):
    population = []
    for i in range(pop_size):
        individual = []
        for j in range(len(bounds)):
            individual.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(individual)
    return population


def evaluate_fitness(population):
    fitness = []
    for individual in population:
        fitness.append(objective_function(individual))
    return np.array(fitness)


def select_parents(population, fitness):
    parent_1, parent_2 = random.choices(
        population, weights=[x**(-3) for x in fitness], k=2)
    return parent_1, parent_2


def crossover(parent_1, parent_2):
    crossover_point = random.randint(0, len(parent_1) - 1)
    offspring = parent_1[:crossover_point] + parent_2[crossover_point:]
    return offspring


def mutation(offspring, bounds, mutation_rate):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i] += random.uniform(-0.1, 0.1)
            while offspring[i] < bounds[i][0] or offspring[i] > bounds[i][1]:
                offspring[i] += random.uniform(-0.1, 0.1)
        else:
            if random.random() < mutation_rate:
                offspring[i] = random.uniform(bounds[i][0], bounds[i][1])
    return offspring


def point2(T4, RH4):
    Q4 = 8.333333333
    Qsr = 100
    Qlr = 30
    Qvent = 70
    T1 = 33.2 + 273.15
    T2_goal = 25 + 273.15
    Qvent_s = 1.1 * 6000 * (T1 - T2_goal) * 0.58857777021102
    Qvent_s = Qvent_s * 2.931 * 10 ** -4
    Qvent_l = Qvent - Qvent_s
    Qst = Qsr + Qvent_s
    Qlt = Qlr + Qvent_l
    v4 = coolprop.HAPropsSI('V', 'T', T4, 'P', 101325, 'R', RH4)
    m4_dot = Q4/v4
    T2 = (Qst/m4_dot) + T4
    w4 = coolprop.HAPropsSI('W', 'T', T4, 'P', 101325, 'R', RH4)
    hlr = coolprop.HAPropsSI('H', 'T', T2, 'P', 101325, 'W', w4)/1000
    h2 = (Qlt/m4_dot) + hlr
    RH2 = coolprop.HAPropsSI('R', 'T', T2, 'P', 101325, 'H', h2*1000)
    return T2, RH2


population = generate_population(bounds, pop_size)
printFormat = "Generation {}: Best Individual = {}, Fitness = {}"


def generate_new_population(bounds, pop_size, mutation_rate):
    new_population = []
    while len(new_population) < pop_size:
        parent_1, parent_2 = select_parents(population, fitness)
        offspring = crossover(parent_1, parent_2)
        offspring = mutation(offspring, bounds, mutation_rate)
        new_population.append(offspring)
    return new_population


for i in range(num_generations):
    fitness = evaluate_fitness(population)
    best_individual = population[np.argmin(fitness)]
    print(printFormat.format(i, best_individual, min(fitness)))
    population = generate_new_population(bounds, pop_size, 0.1)
