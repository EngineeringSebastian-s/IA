import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. PREPARACIÓN DE DATOS ---

# Cargar datos (Asegúrate de que la ruta sea correcta)
try:
    df = pd.read_csv("data/IoTData --Raw--.csv")
except FileNotFoundError:
    # Fallback por si no encuentra el archivo al ejecutar la demo
    from io import StringIO

    csv_data = """
    "id","timestamp","pH","TDS","water_level","DHT_temp","DHT_humidity","water_temp","pH_reducer","add_water","nutrients_adder","humidifier","ex_fan"
    "1","2023-11-26 10:57:52","7","500","0","25.5","60","20","ON",,"OFF","OFF","ON"
    """
    df = pd.read_csv(StringIO(csv_data))

# --- LIMPIEZA ---
cols_actuadores = ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]
# Verificar si las columnas existen antes de limpiar
existing_cols = [c for c in cols_actuadores if c in df.columns]
if existing_cols:
    df[existing_cols] = df[existing_cols].fillna("OFF")
    mapping = {'ON': 1, 'OFF': 0}
    for col in existing_cols:
        df[col] = df[col].map(mapping)

# --- AUMENTO DE DATOS (Simulación) ---
np.random.seed(42)
synthetic_data = []
for _ in range(100):
    temp = np.random.uniform(20, 35)
    hum = np.random.uniform(40, 90)
    fan = 1 if (temp > 28 or hum > 75) else 0

    ph = np.random.uniform(5.0, 8.0)
    tds = np.random.uniform(300, 800)
    reducer = 1 if ph > 6.2 else 0

    synthetic_data.append([ph, tds, temp, hum, reducer, fan])

df_synth = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])

# --- ENTRENAMIENTO MODELOS ---

# Modelo 1: Regresión Logística (Ventilador)
X_log = df_synth[['DHT_temp', 'DHT_humidity']]
y_log = df_synth['ex_fan']
log_reg = LogisticRegression()
log_reg.fit(X_log, y_log)

# Modelo 2: Red Neuronal (pH)
X_nn = df_synth[['pH', 'TDS']]
y_nn = df_synth['pH_reducer']
scaler = StandardScaler()
X_nn_scaled = scaler.fit_transform(X_nn)
nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)
nn_model.fit(X_nn_scaled, y_nn)


# --- INTERFAZ GRÁFICA ---

class SmartPotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartPot Dashboard - Control Inteligente")
        self.root.geometry("1000x650")

        # Estilos
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11, "bold"))

        # --- Panel Izquierdo: Controles (Perillas) ---
        input_frame = ttk.LabelFrame(root, text="Panel de Control Manual (Simulación)", padding=20)
        input_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        # Función auxiliar para crear perillas (sliders)
        def create_slider(parent, label_text, min_val, max_val, default_val, resolution, row):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=(10, 0))

            # Usamos tk.Scale nativo porque permite visualizar el valor numérico fácilmente
            scale = tk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                             resolution=resolution, length=250,
                             activebackground="#4CAF50", highlightthickness=0)
            scale.set(default_val)
            scale.grid(row=row + 1, column=0, pady=(0, 10))
            return scale

        # 1. Slider Temperatura
        self.scale_temp = create_slider(input_frame, "🌡️ Temperatura (°C):", 10, 40, 25.5, 0.1, 0)

        # 2. Slider Humedad
        self.scale_hum = create_slider(input_frame, "💧 Humedad (%):", 0, 100, 60, 1, 2)

        # 3. Slider pH
        self.scale_ph = create_slider(input_frame, "⚗️ pH del Agua:", 0, 14, 7.0, 0.01, 4)

        # 4. Slider TDS
        self.scale_tds = create_slider(input_frame, "🧂 TDS (ppm):", 0, 1000, 500, 10, 6)

        # Botón Predecir
        btn_predict = ttk.Button(input_frame, text="ANALIZAR DATOS", command=self.predict_actions)
        btn_predict.grid(row=8, column=0, pady=20, sticky="ew")

        # Resultados
        self.result_frame = ttk.LabelFrame(root, text="Decisión de la IA", padding=20)
        self.result_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

        self.lbl_fan = ttk.Label(self.result_frame, text="Ventilador: ---", font=("Segoe UI", 12))
        self.lbl_fan.pack(anchor="w", pady=5)

        self.lbl_ph = ttk.Label(self.result_frame, text="Reductor pH: ---", font=("Segoe UI", 12))
        self.lbl_ph.pack(anchor="w", pady=5)

        # --- Panel Derecho: Gráficas ---
        self.graph_frame = ttk.Frame(root)
        self.graph_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        self.plot_graphs()

    def predict_actions(self):
        try:
            # 1. Obtener valores DIRECTAMENTE de los Sliders (ya son float/int)
            t = self.scale_temp.get()
            h = self.scale_hum.get()
            ph = self.scale_ph.get()
            tds = self.scale_tds.get()

            # --- Crear DataFrames con nombres CORRECTOS ---
            input_log_df = pd.DataFrame([[t, h]], columns=['DHT_temp', 'DHT_humidity'])
            input_nn_df = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

            # 2. Predicciones
            fan_pred = log_reg.predict(input_log_df)[0]
            fan_prob = log_reg.predict_proba(input_log_df)[0][1]

            input_nn_scaled = scaler.transform(input_nn_df)
            ph_pred = nn_model.predict(input_nn_scaled)[0]

            # 3. Actualizar UI con colores visuales
            if fan_pred == 1:
                self.lbl_fan.config(text=f"💨 Ventilador: ACTIVADO ({fan_prob * 100:.1f}%)", foreground="red")
            else:
                self.lbl_fan.config(text=f"💨 Ventilador: APAGADO ({fan_prob * 100:.1f}%)", foreground="green")

            if ph_pred == 1:
                self.lbl_ph.config(text="🧪 Reductor pH: DOSIFICAR", foreground="red")
            else:
                self.lbl_ph.config(text="🧪 Reductor pH: ESTABLE", foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def plot_graphs(self):
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), dpi=90)

        # Gráfico 1
        scatter = ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'], c=df_synth['ex_fan'], cmap='coolwarm',
                              alpha=0.7)
        ax1.set_title("Regresión Logística: Zona de Activación Ventilador")
        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Humedad (%)")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Leyenda manual simple
        legend_elements = scatter.legend_elements()
        ax1.legend(legend_elements[0], ["Apagado", "Encendido"], loc="upper left")

        # Gráfico 2
        ax2.plot(nn_model.loss_curve_, color='purple', linewidth=2)
        ax2.set_title("Red Neuronal: Curva de Aprendizaje (Error vs Tiempo)")
        ax2.set_xlabel("Iteraciones de Entrenamiento")
        ax2.set_ylabel("Pérdida (Loss)")
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Canvas
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


# Ejecutar App
if __name__ == "__main__":
    root = tk.Tk()
    # Forzar tema para que se vea un poco más moderno
    try:
        root.tk.call("source", "azure.tcl")  # Opcional si tienes temas
        root.tk.call("set_theme", "light")
    except:
        pass

    app = SmartPotApp(root)
    root.mainloop()