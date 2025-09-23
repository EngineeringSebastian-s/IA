import random
import math

# --- Parámetros ---
GRAVEDAD = 9.81
MASA = 80
AREA_CUERPO = 0.5
AREA_PARACAIDAS = 10  # Un paracaídas desafiante
COEF_ARRASTRE = 1.0
DENSIDAD_AIRE = 1.225
ALTITUD_INICIAL = 10000
VELOCIDAD_INICIAL = 0
DT = 0.1
VELOCIDAD_ATERRIZAJE_SEGURA = 35  # Ajustado: desafiante pero posible (~11-50 m/s rango)

# --- ALGORITMO GENÉTICO ---
TAMANO_POBLACION = 100
NUM_GENERACIONES = 100
PROBABILIDAD_CRUCE = 0.95
PROBABILIDAD_MUTACION = 0.2  # Bajada para estabilidad en espacio continuo

# --- Simulación y Fitness ---
def simular_caida(altitud_apertura):
    altitud = ALTITUD_INICIAL
    velocidad = VELOCIDAD_INICIAL
    paracaidas_abierto = False
    while altitud > 0:
        if not paracaidas_abierto and altitud <= altitud_apertura:
            paracaidas_abierto = True
        area_actual = AREA_PARACAIDAS if paracaidas_abierto else AREA_CUERPO
        fuerza_gravedad = MASA * GRAVEDAD
        magnitud_arrastre = 0.5 * DENSIDAD_AIRE * (velocidad**2) * COEF_ARRASTRE * area_actual
        fuerza_neta = fuerza_gravedad - math.copysign(1.0, velocidad) * magnitud_arrastre
        aceleracion = fuerza_neta / MASA
        velocidad += aceleracion * DT
        altitud -= velocidad * DT
        if altitud <= 0 and not paracaidas_abierto:
            return 999
    # Clamp para evitar overshoot
    if altitud < 0:
        altitud = 0
    return abs(velocidad)  # Abs para seguridad

def calcular_fitness(individuo):
    altitud_apertura = individuo
    if altitud_apertura <= 0 or altitud_apertura > ALTITUD_INICIAL:
        return 0.0
    velocidad_final = simular_caida(altitud_apertura)
    error = abs(velocidad_final - VELOCIDAD_ATERRIZAJE_SEGURA)
    return 1 / (1 + error)

def seleccion_ruleta(poblacion_con_fitness):
    """Selecciona un individuo usando el método de la Ruleta."""
    suma_total_fitness = sum(fitness for individuo, fitness in poblacion_con_fitness)
    if suma_total_fitness == 0:
        return random.choice(poblacion_con_fitness)[0]
    punto_seleccion = random.uniform(0, suma_total_fitness)
    fitness_acumulado = 0
    for individuo, fitness in poblacion_con_fitness:
        fitness_acumulado += fitness
        if fitness_acumulado >= punto_seleccion:
            return individuo
    return poblacion_con_fitness[-1][0]

# --- Operadores de Cruce y Mutación ---
def cruce_aritmetico(padre1, padre2):
    hijo = (padre1 + padre2) / 2
    return hijo

def mutacion_gaussiana(individuo):
    mutacion = random.gauss(0, 200)  # σ=200 para variabilidad en altitud (0-4000)
    individuo_mutado = individuo + mutacion
    return max(1, min(individuo_mutado, ALTITUD_INICIAL))

# --- FLUJO GENERAL ---
def ejecutar_algoritmo_genetico():
    poblacion = [random.uniform(1, ALTITUD_INICIAL) for _ in range(TAMANO_POBLACION)]
    print("--- Optimizando aterrizaje del paracaidista ---")
    mejor_solucion_global = (None, 0.0)

    for gen in range(NUM_GENERACIONES):
        poblacion_con_fitness = [(ind, calcular_fitness(ind)) for ind in poblacion]
        mejor_de_la_generacion = max(poblacion_con_fitness, key=lambda item: item[1])

        if mejor_de_la_generacion[1] > mejor_solucion_global[1]:
            mejor_solucion_global = mejor_de_la_generacion

        # CORRECCIÓN: Calcular y print DESPUÉS de actualizar el mejor
        velocidad_resultante = simular_caida(mejor_solucion_global[0])

        # Print cada 10 gens para menos spam
        if (gen + 1) % 10 == 0 or gen == 0:
            print(
                f"Gen {gen+1:3d}: Mejor Altitud = {mejor_solucion_global[0]:7.2f} m | "
                f"Velocidad Final = {velocidad_resultante:5.2f} m/s | "
                f"Fitness = {mejor_solucion_global[1]:.4f}"
            )

        if mejor_solucion_global[1] > 0.98:
            print("\n¡Solución suficientemente buena encontrada!")
            break

        # Elitismo: Mantener top 2
        elites = sorted(poblacion_con_fitness, key=lambda x: x[1], reverse=True)[:2]
        nueva_poblacion = [elite[0] for elite in elites]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre1 = seleccion_ruleta(poblacion_con_fitness)
            padre2 = seleccion_ruleta(poblacion_con_fitness)
            
            if random.random() < PROBABILIDAD_CRUCE:
                hijo = cruce_aritmetico(padre1, padre2)
            else:
                hijo = padre1
            
            if random.random() < PROBABILIDAD_MUTACION:
                hijo = mutacion_gaussiana(hijo)
                
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion

    print("\n--- Simulación Final con la Mejor Solución ---")
    mejor_individuo_final = mejor_solucion_global[0]
    velocidad_final_optima = simular_caida(mejor_individuo_final)
    print(f"La altitud de apertura óptima encontrada es: {mejor_individuo_final:.2f} metros.")
    print(f"Esto resulta en una velocidad de aterrizaje de: {velocidad_final_optima:.2f} m/s.")
    print(f"(Objetivo: {VELOCIDAD_ATERRIZAJE_SEGURA:.2f} m/s)")

if __name__ == "__main__":
    ejecutar_algoritmo_genetico()