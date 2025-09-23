import random
import math
import matplotlib.pyplot as plt

# ---------------------------
# Configuración
# ---------------------------
random_seed = 42
random.seed(random_seed)

population_size = 30
max_generations = 200
tournament_size = 3
elitism = 2
mutation_rate = 0.1
crossover_rate = 0.8

DISPLAY_GENERATIONS = [1, 10, 100, 200]

# Objetivo (paracaidista debe aterrizar en x=0)
TARGET_X = 0
TARGET_SPEED = 0

# ---------------------------
# Representación de un individuo
# ---------------------------
def create_individual():
    # [opening_altitude, wind_correction, descent_angle]
    return [
        random.uniform(500, 2000),  # apertura entre 500 y 2000m
        random.uniform(-50, 50),   # corrección viento en metros
        random.uniform(-30, 30)    # ángulo de caída en grados
    ]

# ---------------------------
# Fitness: qué tan cerca del objetivo aterriza
# ---------------------------
def fitness(ind):
    opening_altitude, wind_correction, descent_angle = ind

    # Modelo simplificado del aterrizaje
    landing_x = wind_correction + descent_angle * 2  # desplazamiento final
    landing_speed = max(0, 5 - (opening_altitude / 500))  # menor altura => más rápido

    # Distancia al objetivo
    distance_error = abs(landing_x - TARGET_X)
    speed_error = abs(landing_speed - TARGET_SPEED)

    return 1 / (1 + distance_error + speed_error)

# ---------------------------
# Selección por torneo
# ---------------------------
def tournament_selection(pop):
    tournament = random.sample(pop, tournament_size)
    return max(tournament, key=lambda ind: fitness(ind))

# ---------------------------
# Cruce
# ---------------------------
def crossover(p1, p2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1[:], p2[:]

# ---------------------------
# Mutación
# ---------------------------
def mutate(ind):
    for i in range(len(ind)):
        if random.random() < mutation_rate:
            if i == 0:  # apertura
                ind[i] += random.uniform(-100, 100)
            elif i == 1:  # viento
                ind[i] += random.uniform(-10, 10)
            else:  # ángulo
                ind[i] += random.uniform(-5, 5)
    return ind

# ---------------------------
# Visualización del paracaidista
# ---------------------------
def display_parachutist(ind, gen):
    opening_altitude, wind_correction, descent_angle = ind
    landing_x = wind_correction + descent_angle * 2

    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="black", linestyle="--")  # suelo
    plt.scatter([TARGET_X], [0], color="red", marker="*", s=200, label="Objetivo")
    plt.scatter([landing_x], [0], color="blue", s=100, label="Paracaidista")
    plt.title(f"Generación {gen} | Aterrizaje simulado")
    plt.xlabel("Posición horizontal (x)")
    plt.ylabel("Altura (suelo = 0)")
    plt.legend()
    plt.show()

# ---------------------------
# Algoritmo Genético
# ---------------------------
def genetic_algorithm():
    population = [create_individual() for _ in range(population_size)]
    best_fitness_history = []

    for gen in range(1, max_generations + 1):
        # Evaluar
        scored = sorted(population, key=lambda ind: fitness(ind), reverse=True)
        best_ind = scored[0]
        best_fit = fitness(best_ind)
        best_fitness_history.append(best_fit)

        print(f"Gen {gen} | Mejor fitness: {best_fit:.4f} | Mejor ind: {best_ind}")

        # Mostrar ilustración en generaciones clave
        if gen in DISPLAY_GENERATIONS:
            display_parachutist(best_ind, gen)

        # Elitismo
        new_population = scored[:elitism]

        # Generar nuevos hijos
        while len(new_population) < population_size:
            parent1 = tournament_selection(scored)
            parent2 = tournament_selection(scored)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))

        population = new_population

    best = max(population, key=lambda ind: fitness(ind))
    return best, best_fitness_history

# ---------------------------
# Ejecución
# ---------------------------
best_solution, history = genetic_algorithm()
print("\nMejor solución encontrada:", best_solution, "Fitness:", fitness(best_solution))

# ---------------------------
# Gráfico de evolución del fitness
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(history, label="Mejor fitness por generación", color="green")
plt.title("Evolución del Fitness")
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.show()
