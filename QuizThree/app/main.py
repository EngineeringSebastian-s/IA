import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. PREPARACIÓN DE DATOS --

# Cargar datos
df = pd.read_csv("data/IoTData --Raw--.csv")

# --- LIMPIEZA ---
# 1. Llenar vacíos en actuadores con 'OFF'
cols_actuadores = ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]
df[cols_actuadores] = df[cols_actuadores].fillna("OFF")

# 2. Convertir ON/OFF a 1/0
mapping = {'ON': 1, 'OFF': 0}
for col in cols_actuadores:
    df[col] = df[col].map(mapping)

# --- AUMENTO DE DATOS (Simulación para entrenar mejor) ---
# Como hay pocos datos, generamos ruido sintético para que los modelos aprendan algo útil en la demo
np.random.seed(42)
synthetic_data = []
for _ in range(100):
    # Simulación: Si Temp > 28 o Humedad > 70, encender Fan
    temp = np.random.uniform(20, 35)
    hum = np.random.uniform(40, 90)
    fan = 1 if (temp > 28 or hum > 75) else 0

    # Simulación: Si pH > 6.5, encender Reducer
    ph = np.random.uniform(5.0, 8.0)
    tds = np.random.uniform(300, 800)
    reducer = 1 if ph > 6.2 else 0

    synthetic_data.append([ph, tds, temp, hum, reducer, fan])

df_synth = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])

# --- ENTRENAMIENTO MODELOS ---

# Modelo 1: Regresión Logística (Para el Ventilador)
X_log = df_synth[['DHT_temp', 'DHT_humidity']]
y_log = df_synth['ex_fan']
log_reg = LogisticRegression()
log_reg.fit(X_log, y_log)

# Modelo 2: Red Neuronal (Para el pH)
X_nn = df_synth[['pH', 'TDS']]
y_nn = df_synth['pH_reducer']
# Escalamos datos para la Red Neuronal (muy importante para convergencia)
scaler = StandardScaler()
X_nn_scaled = scaler.fit_transform(X_nn)

# Red neuronal simple: 2 capas ocultas de 5 neuronas
nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)
nn_model.fit(X_nn_scaled, y_nn)


# --- INTERFAZ GRÁFICA (TKINTER) ---

class SmartPotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartPot Dashboard - Control Inteligente")
        self.root.geometry("900x600")

        # Estilos
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 12))
        style.configure("TButton", font=("Helvetica", 11))

        # --- Panel Izquierdo: Entradas ---
        input_frame = ttk.LabelFrame(root, text="Lecturas de Sensores en Tiempo Real", padding=20)
        input_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Entradas
        ttk.Label(input_frame, text="Temperatura (°C):").grid(row=0, column=0, sticky="w")
        self.entry_temp = ttk.Entry(input_frame)
        self.entry_temp.insert(0, "25.5")
        self.entry_temp.grid(row=0, column=1, pady=5)

        ttk.Label(input_frame, text="Humedad (%):").grid(row=1, column=0, sticky="w")
        self.entry_hum = ttk.Entry(input_frame)
        self.entry_hum.insert(0, "60")
        self.entry_hum.grid(row=1, column=1, pady=5)

        ttk.Label(input_frame, text="pH Actual:").grid(row=2, column=0, sticky="w")
        self.entry_ph = ttk.Entry(input_frame)
        self.entry_ph.insert(0, "7.0")
        self.entry_ph.grid(row=2, column=1, pady=5)

        ttk.Label(input_frame, text="TDS (ppm):").grid(row=3, column=0, sticky="w")
        self.entry_tds = ttk.Entry(input_frame)
        self.entry_tds.insert(0, "500")
        self.entry_tds.grid(row=3, column=1, pady=5)

        # Botón Predecir
        btn_predict = ttk.Button(input_frame, text="Analizar y Sugerir Acciones", command=self.predict_actions)
        btn_predict.grid(row=4, column=0, columnspan=2, pady=20, sticky="ew")

        # Resultados
        self.result_frame = ttk.LabelFrame(root, text="Estado de Actuadores (IA)", padding=20)
        self.result_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nw")

        self.lbl_fan = ttk.Label(self.result_frame, text="Ventilador: ---", foreground="gray")
        self.lbl_fan.pack(anchor="w")
        self.lbl_ph = ttk.Label(self.result_frame, text="Reductor pH: ---", foreground="gray")
        self.lbl_ph.pack(anchor="w")

        # --- Panel Derecho: Gráficas ---
        self.graph_frame = ttk.Frame(root)
        self.graph_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.plot_graphs()

    def predict_actions(self):
        try:
            # 1. Obtener valores
            t = float(self.entry_temp.get())
            h = float(self.entry_hum.get())
            ph = float(self.entry_ph.get())
            tds = float(self.entry_tds.get())

            # 2. Predicción Regresión Logística (Ventilador)
            fan_pred = log_reg.predict([[t, h]])[0]
            fan_prob = log_reg.predict_proba([[t, h]])[0][1]  # Probabilidad de ser 1

            # 3. Predicción Red Neuronal (pH)
            # Importante: escalar la entrada igual que en el entrenamiento
            input_nn = scaler.transform([[ph, tds]])
            ph_pred = nn_model.predict(input_nn)[0]

            # 4. Actualizar UI
            if fan_pred == 1:
                self.lbl_fan.config(text=f"Ventilador: ENCENDER (Prob: {fan_prob:.2f})", foreground="red")
            else:
                self.lbl_fan.config(text=f"Ventilador: APAGAR (Prob: {fan_prob:.2f})", foreground="green")

            if ph_pred == 1:
                self.lbl_ph.config(text="Reductor pH: DOSIFICAR", foreground="red")
            else:
                self.lbl_ph.config(text="Reductor pH: ESPERAR", foreground="green")

        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos")

    def plot_graphs(self):
        # Crear figura de matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), dpi=100)

        # Gráfico 1: Dispersión Temp vs Humedad (Coloreado por Ventilador ON/OFF)
        scatter = ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'], c=df_synth['ex_fan'], cmap='bwr',
                              alpha=0.6)
        ax1.set_title("Regresión Logística: Activación Ventilador")
        ax1.set_xlabel("Temperatura")
        ax1.set_ylabel("Humedad")
        ax1.legend(*scatter.legend_elements(), title="Fan (0=Off, 1=On)")

        # Gráfico 2: Curva de pérdida de la Red Neuronal
        ax2.plot(nn_model.loss_curve_)
        ax2.set_title("Red Neuronal: Curva de Aprendizaje")
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Loss (Error)")

        plt.tight_layout()

        # Insertar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


# Ejecutar App
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartPotApp(root)
    root.mainloop()