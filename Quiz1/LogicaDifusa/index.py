import tkinter as tk
from tkinter import messagebox
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 1. Definición del Sistema Difuso (El "Motor")
# Se define una sola vez fuera de la función de la GUI para eficiencia.
# --------------------------------------------------------------------------

# Universos de discurso
universo_tamaño = np.arange(0, 101, 1)
universo_diferencia = np.arange(0, 50, 1)
universo_decision = np.arange(0, 101, 1)

# Variables de entrada (Antecedentes)
tamaño_vela_anterior = ctrl.Antecedent(universo_tamaño, 'tamaño_vela_anterior')
tamaño_vela_actual = ctrl.Antecedent(universo_tamaño, 'tamaño_vela_actual')
diferencia_envolvimiento = ctrl.Antecedent(universo_diferencia, 'diferencia_envolvimiento')

# Variable de salida (Consecuente)
decision_inversion = ctrl.Consequent(universo_decision, 'decision_inversion')

# Funciones de membresía
tamaño_vela_anterior.automf(names=['corta', 'media', 'grande'])
tamaño_vela_actual.automf(names=['corta', 'media', 'grande'])

diferencia_envolvimiento['poco_envolvente'] = fuzz.trimf(universo_diferencia, [0, 0, 25])
diferencia_envolvimiento['envolvente'] = fuzz.trimf(universo_diferencia, [10, 25, 40])
diferencia_envolvimiento['muy_envolvente'] = fuzz.trimf(universo_diferencia, [25, 50, 50])

decision_inversion['invertir_poco'] = fuzz.trimf(universo_decision, [0, 0, 50])
decision_inversion['invertir_normal'] = fuzz.trimf(universo_decision, [25, 50, 75])
decision_inversion['invertir_mucho'] = fuzz.trimf(universo_decision, [50, 100, 100])

# Conjunto de reglas
regla1 = ctrl.Rule(tamaño_vela_anterior['corta'] & tamaño_vela_actual['grande'] & diferencia_envolvimiento['muy_envolvente'],
                   decision_inversion['invertir_mucho'])
regla2 = ctrl.Rule(tamaño_vela_actual['corta'], decision_inversion['invertir_poco'])
regla3 = ctrl.Rule(tamaño_vela_anterior['media'] & tamaño_vela_actual['media'] & diferencia_envolvimiento['envolvente'],
                   decision_inversion['invertir_normal'])
regla4 = ctrl.Rule(tamaño_vela_actual['media'] & diferencia_envolvimiento['muy_envolvente'],
                   decision_inversion['invertir_mucho'])
regla5 = ctrl.Rule(diferencia_envolvimiento['poco_envolvente'], decision_inversion['invertir_poco'])
regla6 = ctrl.Rule(tamaño_vela_anterior['corta'] & tamaño_vela_actual['media'],
                   decision_inversion['invertir_normal'])
regla7 = ctrl.Rule(tamaño_vela_anterior['grande'] & tamaño_vela_actual['grande'] & diferencia_envolvimiento['poco_envolvente'],
                   decision_inversion['invertir_poco'])

# Sistema de control
sistema_ctrl = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6, regla7])
inversion_simulador = ctrl.ControlSystemSimulation(sistema_ctrl)


# --------------------------------------------------------------------------
# 2. Función que se ejecuta al presionar el botón "Calcular"
# --------------------------------------------------------------------------
def calcular_decision():
    """
    Toma los valores de la GUI, ejecuta el sistema difuso y muestra los resultados.
    """
    try:
        # Obtener valores de los campos de entrada
        valor_anterior = float(entry_anterior.get())
        valor_actual = float(entry_actual.get())
        valor_diferencia = float(entry_diferencia.get())

        # Validar rangos (opcional pero recomendado)
        if not (0 <= valor_anterior <= 100 and 0 <= valor_actual <= 100):
            messagebox.showerror("Error de Rango", "El tamaño de las velas debe estar entre 0 y 100.")
            return
        if not (0 <= valor_diferencia <= 50):
            messagebox.showerror("Error de Rango", "La diferencia de envolvimiento debe estar entre 0 y 50.")
            return

    except ValueError:
        messagebox.showerror("Error de Entrada", "Por favor, ingrese solo valores numéricos.")
        return

    # --- PASO 1: FUSIFICACIÓN ---
    memb_anterior_corta = fuzz.interp_membership(universo_tamaño, tamaño_vela_anterior['corta'].mf, valor_anterior)
    memb_anterior_media = fuzz.interp_membership(universo_tamaño, tamaño_vela_anterior['media'].mf, valor_anterior)
    memb_anterior_grande = fuzz.interp_membership(universo_tamaño, tamaño_vela_anterior['grande'].mf, valor_anterior)

    memb_actual_corta = fuzz.interp_membership(universo_tamaño, tamaño_vela_actual['corta'].mf, valor_actual)
    memb_actual_media = fuzz.interp_membership(universo_tamaño, tamaño_vela_actual['media'].mf, valor_actual)
    memb_actual_grande = fuzz.interp_membership(universo_tamaño, tamaño_vela_actual['grande'].mf, valor_actual)

    memb_dif_poco = fuzz.interp_membership(universo_diferencia, diferencia_envolvimiento['poco_envolvente'].mf, valor_diferencia)
    memb_dif_normal = fuzz.interp_membership(universo_diferencia, diferencia_envolvimiento['envolvente'].mf, valor_diferencia)
    memb_dif_muy = fuzz.interp_membership(universo_diferencia, diferencia_envolvimiento['muy_envolvente'].mf, valor_diferencia)

    # --- PASO 2: ACTIVACIÓN DE REGLAS ---
    fuerza_regla1 = np.fmin(memb_anterior_corta, np.fmin(memb_actual_grande, memb_dif_muy))
    fuerza_regla2 = memb_actual_corta
    fuerza_regla3 = np.fmin(memb_anterior_media, np.fmin(memb_actual_media, memb_dif_normal))
    fuerza_regla4 = np.fmin(memb_actual_media, memb_dif_muy)
    fuerza_regla5 = memb_dif_poco
    fuerza_regla6 = np.fmin(memb_anterior_corta, memb_actual_media)
    fuerza_regla7 = np.fmin(memb_anterior_grande, np.fmin(memb_actual_grande, memb_dif_poco))

    # --- PASO FINAL: CÁLCULO DEL SIMULADOR ---
    inversion_simulador.input['tamaño_vela_anterior'] = valor_anterior
    inversion_simulador.input['tamaño_vela_actual'] = valor_actual
    inversion_simulador.input['diferencia_envolvimiento'] = valor_diferencia
    inversion_simulador.compute()
    decision_final = inversion_simulador.output['decision_inversion']

    # --- Formatear la cadena de resultados ---
    output_string = "--- PASO 1: FUSIFICACIÓN DE ENTRADAS ---\n\n"
    output_string += f"Grado de membresía para tamaño_vela_anterior ({valor_anterior}):\n"
    output_string += f"  - corta: {memb_anterior_corta:.2f}\n"
    output_string += f"  - media: {memb_anterior_media:.2f}\n"
    output_string += f"  - grande: {memb_anterior_grande:.2f}\n\n"
    output_string += f"Grado de membresía para tamaño_vela_actual ({valor_actual}):\n"
    output_string += f"  - corta: {memb_actual_corta:.2f}\n"
    output_string += f"  - media: {memb_actual_media:.2f}\n"
    output_string += f"  - grande: {memb_actual_grande:.2f}\n\n"
    output_string += f"Grado de membresía para diferencia_envolvimiento ({valor_diferencia}):\n"
    output_string += f"  - poco_envolvente: {memb_dif_poco:.2f}\n"
    output_string += f"  - envolvente: {memb_dif_normal:.2f}\n"
    output_string += f"  - muy_envolvente: {memb_dif_muy:.2f}\n\n"

    output_string += "--- PASO 2: ACTIVACIÓN DE REGLAS (INFERENCIA) ---\n\n"
    output_string += f"Fuerza de activación de cada regla:\n"
    output_string += f"  - Regla 1 (invertir_mucho): {fuerza_regla1:.2f}\n"
    output_string += f"  - Regla 2 (invertir_poco): {fuerza_regla2:.2f}\n"
    output_string += f"  - Regla 3 (invertir_normal): {fuerza_regla3:.2f}\n"
    output_string += f"  - Regla 4 (invertir_mucho): {fuerza_regla4:.2f}\n"
    output_string += f"  - Regla 5 (invertir_poco): {fuerza_regla5:.2f}\n"
    output_string += f"  - Regla 6 (invertir_normal): {fuerza_regla6:.2f}\n"
    output_string += f"  - Regla 7 (invertir_poco): {fuerza_regla7:.2f}\n\n"

    output_string += "--- PASO FINAL: RESULTADO DE LA DEFUSIFICACIÓN ---\n\n"
    output_string += f"Valor de decisión de inversión: {decision_final:.2f}\n"

    # --- Mostrar resultados en el widget de texto ---
    results_text.config(state=tk.NORMAL) # Habilitar para escribir
    results_text.delete('1.0', tk.END) # Limpiar contenido anterior
    results_text.insert(tk.END, output_string)
    results_text.config(state=tk.DISABLED) # Deshabilitar para que sea de solo lectura

    # --- Mostrar la gráfica ---
    decision_inversion.view(sim=inversion_simulador)
    plt.suptitle("Gráfico de Decisión Final")
    plt.show()


# --------------------------------------------------------------------------
# 3. Creación y configuración de la ventana principal de Tkinter
# --------------------------------------------------------------------------
window = tk.Tk()
window.title("Calculadora de Patrón Envolvente Alcista")
window.geometry("650x700") # Tamaño de la ventana

# Crear un frame para los inputs
input_frame = tk.Frame(window, padx=10, pady=10)
input_frame.pack(fill=tk.X)

# --- Widgets de entrada ---
tk.Label(input_frame, text="Tamaño Vela Anterior (0-100):").grid(row=0, column=0, sticky="w", pady=2)
entry_anterior = tk.Entry(input_frame)
entry_anterior.grid(row=0, column=1, pady=2)

tk.Label(input_frame, text="Tamaño Vela Actual (0-100):").grid(row=1, column=0, sticky="w", pady=2)
entry_actual = tk.Entry(input_frame)
entry_actual.grid(row=1, column=1, pady=2)

tk.Label(input_frame, text="Diferencia Envolvimiento (0-50):").grid(row=2, column=0, sticky="w", pady=2)
entry_diferencia = tk.Entry(input_frame)
entry_diferencia.grid(row=2, column=1, pady=2)

# --- Botón de cálculo ---
calculate_button = tk.Button(window, text="Calcular Decisión", command=calcular_decision)
calculate_button.pack(pady=10)

# --- Widget de texto para mostrar resultados ---
results_text = tk.Text(window, height=25, width=80, wrap=tk.WORD, state=tk.DISABLED)
results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Iniciar el bucle principal de la GUI
window.mainloop()
