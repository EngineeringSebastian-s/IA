import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from copy import deepcopy

# ---------- Parámetros configurables ----------

INPUT_IMAGE_PATH = ""              # Ruta de la imagen objetivo a replicar. Si está vacío, se genera una imagen por defecto
IMAGE_SIZE = (8, 8)                # Tamaño de la imagen a procesar (ancho, alto). Menor tamaño = más rápido
POPULATION_SIZE = 1000             # Número de individuos (soluciones) en cada generación
MAX_GENERATIONS = 1000            # Número máximo de generaciones del algoritmo genético
TOURNAMENT_SIZE = 3               # Número de individuos seleccionados aleatoriamente para competir en el torneo de selección
ELITISM = 2                       # Número de mejores individuos que pasan directamente a la siguiente generación (sin modificación)
MUTATION_RATE = 0.7               # Probabilidad de que un individuo sufra mutación (entre 0 y 1)
CROSSOVER_RATE = 0.9              # Probabilidad de que ocurra cruce entre dos padres (entre 0 y 1)
RANDOM_SEED = 42                  # Semilla fija para garantizar reproducibilidad en los resultados
DISPLAY_GENERATIONS = [1, 10, 100, 1000, 10000]  # Generaciones en las que se guardan o visualizan resultados intermedios
RESTART_INTERVAL = 500           # Intervalo (en generaciones) para reiniciar parcialmente la población y evitar estancamiento


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- Utilidades de imagen ----------
def rgb_to_luminance(rgb):
    return 0.299 * rgb[...,0] + 0.587 * rgb[...,1] + 0.114 * rgb[...,2]

def clamp_rgb(arr):
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def load_or_create_image(path=None, size=(8,8)):
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        arr = np.array(img, dtype=np.uint8)
    else:
        w,h = size
        arr = np.zeros((h,w,3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                r = int((x / (w-1)) * 255)
                g = int((y / (h-1)) * 255)
                b = int(((x+y) / (w+h-2)) * 255)
                if (x+y) % 3 == 0:
                    r, g, b = (b, r, g)
                arr[y,x] = (r,g,b)
    return arr

# ---------- Fitness ----------
def fitness_individual(individual, target_rgb, target_gray, alpha=1.0, beta=0.9):
    err_rgb = np.abs(individual - target_rgb).sum()
    y_ind = 0.299*individual[...,0] + 0.587*individual[...,1] + 0.114*individual[...,2]
    err_y = np.abs(y_ind - target_gray).sum()
    error_total = alpha * err_rgb + beta * err_y
    return 1.0 / (1.0 + error_total)

def fitness_population(population, target_rgb, target_gray, alpha=1.0, beta=0.9):
    N = population.shape[0]
    fits = np.zeros(N, dtype=np.float64)
    for i in range(N):
        fits[i] = fitness_individual(population[i], target_rgb, target_gray, alpha, beta)
    return fits

# ---------- Operadores genéticos ----------
def tournament_selection(pop, fits, k=3):
    idxs = np.random.choice(len(pop), k, replace=False)
    winner = idxs[np.argmax(fits[idxs])]
    return deepcopy(pop[winner])

def uniform_crossover(parent1, parent2):
    mask = np.random.rand(*parent1.shape[:2]) < 0.5
    mask = mask[...,None]
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

# ---------- Mutaciones ----------
def mutate_small_noise(indiv):
    indiv += np.random.normal(0, 6, size=indiv.shape)
    return indiv

def mutate_large_noise(indiv):
    indiv += np.random.normal(0, 20, size=indiv.shape)
    return indiv

def mutate_channel_shift(indiv):
    ch = np.random.randint(0,3)
    shift = np.random.randint(-40,41)
    indiv[...,ch] += shift
    return indiv

def mutate_pixel_swap(indiv):
    h,w,_ = indiv.shape
    x1,y1 = np.random.randint(0,w), np.random.randint(0,h)
    x2,y2 = np.random.randint(0,w), np.random.randint(0,h)
    tmp = indiv[y1,x1].copy()
    indiv[y1,x1] = indiv[y2,x2]
    indiv[y2,x2] = tmp
    return indiv

def mutate_invert_patch(indiv):
    h,w,_ = indiv.shape
    ph = max(1, int(h*0.25))
    pw = max(1, int(w*0.25))
    y = np.random.randint(0, h-ph+1)
    x = np.random.randint(0, w-pw+1)
    indiv[y:y+ph, x:x+pw] = 255 - indiv[y:y+ph, x:x+pw]
    return indiv

def mutate_random_reset(indiv):
    h,w,_ = indiv.shape
    n = max(1, (h*w)//6)
    for _ in range(n):
        x = np.random.randint(0,w); y = np.random.randint(0,h)
        indiv[y,x] = np.random.randint(0,256, size=3)
    return indiv

def mutate_neighbor_average(indiv):
    h,w,_ = indiv.shape
    n = max(1, (h*w)//8)
    for _ in range(n):
        x = np.random.randint(0,w); y = np.random.randint(0,h)
        xs = slice(max(0,x-1), min(w,x+2))
        ys = slice(max(0,y-1), min(h,y+2))
        region = indiv[ys,xs].astype(np.float32)
        indiv[y,x] = np.mean(region.reshape(-1,3), axis=0)
    return indiv

def mutate_scale_channels(indiv):
    factors = 1.0 + np.random.normal(0, 0.12, size=3)
    indiv *= factors[None,None,:]
    return indiv

def mutate_towards_gray(indiv, gray):
    h,w,_ = indiv.shape
    n = max(1, (h*w)//6)
    for _ in range(n):
        x = np.random.randint(0,w); y = np.random.randint(0,h)
        target_y = gray[y,x]
        current = indiv[y,x].astype(np.float32)
        cur_y = 0.299*current[0] + 0.587*current[1] + 0.114*current[2]
        diff = target_y - cur_y
        indiv[y,x] = current + diff * np.array([0.299, 0.587, 0.114])
    return indiv

def mutate_clamp_and_jitter(indiv):
    indiv[:] = np.clip(indiv, 0, 255)
    h,w,_ = indiv.shape
    n = max(1, (h*w)//10)
    for _ in range(n):
        x = np.random.randint(0,w); y = np.random.randint(0,h)
        indiv[y,x] = np.clip(indiv[y,x] + np.random.randint(-8,9,3), 0, 255)
    return indiv

MUTATION_OPERATORS = [
    mutate_small_noise, mutate_large_noise, mutate_channel_shift, mutate_pixel_swap,
    mutate_invert_patch, mutate_random_reset, mutate_neighbor_average, mutate_scale_channels,
    mutate_towards_gray, mutate_clamp_and_jitter
]

# ---------- Inicialización ----------
def initialize_population(pop_size, h, w, gray_image):
    base = np.stack([gray_image, gray_image, gray_image], axis=-1).astype(np.float32)
    population = np.zeros((pop_size, h, w, 3), dtype=np.float32)
    for i in range(pop_size):
        noise = np.random.normal(0, 20.0, size=base.shape)
        population[i] = np.clip(base + noise, 0, 255)
    return population

def reset_population(population, reset_rate=0.2):
    n = int(len(population) * reset_rate)
    for i in range(n):
        h, w, _ = population[i].shape
        population[i] = np.random.rand(h, w, 3) * 255
    return population

# ---------- Algoritmo principal ----------
def run_ga(target_rgb, gray_image, pop_size=100, max_gen=1000):
    h,w,_ = target_rgb.shape
    population = initialize_population(pop_size, h, w, gray_image)
    best_history = []
    best_images = {}
    best_fitness = -1
    best_ind = None

    for gen in range(1, max_gen+1):
        fits = fitness_population(population, target_rgb, gray_image)
        idx_best = np.argmax(fits)
        if fits[idx_best] > best_fitness:
            best_fitness = fits[idx_best]
            best_ind = deepcopy(population[idx_best])
        best_history.append(best_fitness)

        if gen in DISPLAY_GENERATIONS:
            best_images[gen] = clamp_rgb(best_ind.copy())

        # Reinicio parcial
        if gen % RESTART_INTERVAL == 0:
            population = reset_population(population)

        # Nueva generación
        new_pop = []
        elite_idxs = np.argsort(fits)[-ELITISM:]
        for ei in elite_idxs:
            new_pop.append(deepcopy(population[ei]))

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fits, k=TOURNAMENT_SIZE)
            p2 = tournament_selection(population, fits, k=TOURNAMENT_SIZE)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            for child in (c1, c2):
                if random.random() < MUTATION_RATE:
                    op = random.choice(MUTATION_OPERATORS)
                    if op is mutate_towards_gray:
                        child = op(child, gray_image)
                    else:
                        child = op(child)
                child = np.clip(child, 0, 255)
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break

        population = np.array(new_pop, dtype=np.float32)

        if gen % 50 == 0 or gen <= 5:
            print(f"Generación {gen:4d} - Mejor fitness: {best_fitness:.8f}")

    best_images['final'] = clamp_rgb(best_ind.copy())
    return best_history, best_images, clamp_rgb(best_ind.copy())

# ---------- Ejecución ----------
target_rgb = load_or_create_image(INPUT_IMAGE_PATH, size=IMAGE_SIZE)
h,w,_ = target_rgb.shape
target_gray = rgb_to_luminance(target_rgb).astype(np.float32)

print("Tamaño imagen:", target_rgb.shape)
print("Población:", POPULATION_SIZE, "Generaciones:", MAX_GENERATIONS)

history, snapshots, best_final = run_ga(target_rgb.astype(np.float32), target_gray,
                                        pop_size=POPULATION_SIZE, max_gen=MAX_GENERATIONS)

def show_image_grid(images, titles, figsize=(12,6)):
    n = len(images)
    plt.figure(figsize=figsize)
    for i,img in enumerate(images):
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(titles[i])
    plt.show()

gray_img_vis = np.stack([target_gray, target_gray, target_gray], axis=-1).astype(np.uint8)
show_image_grid([target_rgb, gray_img_vis], ["Original (RGB)", "Grayscale (NTSC)"], figsize=(6,3))

snap_gens = [g for g in DISPLAY_GENERATIONS if g in snapshots]
imgs = [snapshots[g] for g in snap_gens]
titles = [f"Gen {g}" for g in snap_gens]
if imgs:
    show_image_grid(imgs, titles, figsize=(3*len(imgs),3))

show_image_grid([best_final], ["Mejor Final"], figsize=(3,3))

plt.figure(figsize=(8,3))
plt.plot(history)
plt.xlabel("Generación")
plt.ylabel("Mejor fitness")
plt.title("Evolución del mejor fitness")
plt.grid(True)
plt.show()
