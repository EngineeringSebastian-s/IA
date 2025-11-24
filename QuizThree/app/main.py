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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

# ==========================================
# 1. GENERACIÓN DE DATOS Y ENTRENAMIENTO
# ==========================================

# Generamos datos sintéticos robustos (500 muestras)
np.random.seed(42)
synthetic_data = []
for _ in range(500):
    temp = np.random.uniform(15, 40)
    hum = np.random.uniform(30, 95)
    # Lógica: Ventilador ON si Temp > 28 OR Humedad > 75 (con ruido)
    fan = 1 if (temp > 28 or hum > 75) else 0
    if np.random.rand() > 0.95: fan = 1 - fan

    ph = np.random.uniform(4.0, 9.0)
    tds = np.random.uniform(200, 900)
    # Lógica: Reductor ON si pH > 6.5 (con ruido)
    reducer = 1 if ph > 6.5 else 0
    if np.random.rand() > 0.95: reducer = 1 - reducer

    synthetic_data.append([ph, tds, temp, hum, reducer, fan])

df_synth = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])

# --- MODELO A: Regresión Logística (Ventilador) ---
X_log = df_synth[['DHT_temp', 'DHT_humidity']]
y_log = df_synth['ex_fan']
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train_log, y_train_log)
y_pred_log = log_reg.predict(X_test_log)

# --- MODELO B: Red Neuronal (pH) ---
X_nn = df_synth[['pH', 'TDS']]
y_nn = df_synth['pH_reducer']
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_nn_scaled = scaler.fit_transform(X_train_nn)
X_test_nn_scaled = scaler.transform(X_test_nn)

nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, random_state=42)
nn_model.fit(X_train_nn_scaled, y_train_nn)
y_pred_nn = nn_model.predict(X_test_nn_scaled)

# Imprimir métricas en consola (opcional, para referencia técnica)
print(f"Precisión Logística: {accuracy_score(y_test_log, y_pred_log):.2f}")
print(f"Precisión Red Neuronal: {accuracy_score(y_test_nn, y_pred_nn):.2f}")


# ==========================================
# 2. INTERFAZ GRÁFICA (DASHBOARD)
# ==========================================

class SmartPotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartPot - Dashboard de Control y Diagnóstico")
        self.root.geometry("1200x700")

        # Estilos visuales
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 11))

        # --- LAYOUT PRINCIPAL ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ---------------------------------------
        # COLUMNA IZQUIERDA: CONTROLES MANUALES
        # ---------------------------------------
        control_panel = ttk.LabelFrame(main_frame, text=" Simulador de Sensores ", padding=15)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Sliders usando la función helper
        self.create_slider(control_panel, "🌡️ Temperatura (°C)", 10, 40, 25.5, 0.1, 0, 'scale_temp')
        self.create_slider(control_panel, "💧 Humedad (%)", 0, 100, 60, 1, 2, 'scale_hum')
        ttk.Separator(control_panel, orient='horizontal').grid(row=4, column=0, sticky="ew", pady=15)
        self.create_slider(control_panel, "⚗️ pH del Agua", 0, 14, 7.0, 0.01, 5, 'scale_ph')
        self.create_slider(control_panel, "🧂 TDS (ppm)", 0, 1000, 500, 10, 7, 'scale_tds')

        # Botón de Acción
        btn_predict = ttk.Button(control_panel, text="ANALIZAR CONDICIONES", command=self.predict_actions)
        btn_predict.grid(row=9, column=0, pady=25, sticky="ew")

        # Panel de Resultados
        self.res_frame = ttk.LabelFrame(control_panel, text=" Estado del Sistema (IA) ", padding=15)
        self.res_frame.grid(row=10, column=0, sticky="ew")

        self.lbl_fan = ttk.Label(self.res_frame, text="Ventilador: ---", style="Result.TLabel", foreground="gray")
        self.lbl_fan.pack(anchor="w", pady=5)
        self.lbl_ph = ttk.Label(self.res_frame, text="Reductor pH: ---", style="Result.TLabel", foreground="gray")
        self.lbl_ph.pack(anchor="w", pady=5)

        # ---------------------------------------
        # COLUMNA DERECHA: VISUALIZACIONES (TABS)
        # ---------------------------------------
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Crear sistema de pestañas
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña 1: Gráficos de Comportamiento
        self.tab_behavior = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_behavior, text="Dinámica del Modelo")

        # Pestaña 2: Matrices de Confusión
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Evaluación de Precisión")

        # Renderizar gráficos
        self.plot_behavior_charts(self.tab_behavior)
        self.plot_confusion_matrices(self.tab_metrics)

    def create_slider(self, parent, text, vmin, vmax, vdef, res, row, attr_name):
        """Helper para crear sliders consistentes"""
        ttk.Label(parent, text=text, style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=(5, 0))
        scale = tk.Scale(parent, from_=vmin, to=vmax, orient=tk.HORIZONTAL, resolution=res, length=240,
                         activebackground="#2196F3", highlightthickness=0)
        scale.set(vdef)
        scale.grid(row=row + 1, column=0, pady=(0, 10))
        setattr(self, attr_name, scale)

    def predict_actions(self):
        try:
            # 1. Obtener valores
            t = self.scale_temp.get()
            h = self.scale_hum.get()
            ph = self.scale_ph.get()
            tds = self.scale_tds.get()

            # 2. Crear DataFrames (Evita warnings de sklearn)
            in_log = pd.DataFrame([[t, h]], columns=['DHT_temp', 'DHT_humidity'])
            in_nn = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

            # 3. Predicciones
            fan_act = log_reg.predict(in_log)[0]
            fan_prob = log_reg.predict_proba(in_log)[0][1]

            in_nn_scaled = scaler.transform(in_nn)
            ph_act = nn_model.predict(in_nn_scaled)[0]

            # 4. Actualizar UI
            # Lógica Ventilador
            if fan_act == 1:
                self.lbl_fan.config(text=f"💨 Ventilador: ACTIVADO ({fan_prob * 100:.1f}%)", foreground="red")
            else:
                self.lbl_fan.config(text=f"💨 Ventilador: APAGADO ({fan_prob * 100:.1f}%)", foreground="green")

            # Lógica pH
            if ph_act == 1:
                self.lbl_ph.config(text="🧪 Reductor pH: DOSIFICAR", foreground="red")
            else:
                self.lbl_ph.config(text="🧪 Reductor pH: ESTABLE", foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo: {e}")

    def plot_behavior_charts(self, parent_frame):
        """Gráficos: Scatter Plot y Curva de Pérdida"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100)

        # A. Scatter Plot (Regresión Logística)
        scatter = ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'],
                              c=df_synth['ex_fan'], cmap='coolwarm', alpha=0.6, edgecolors='w')
        ax1.set_title("Distribución: Temperatura vs Humedad")
        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Humedad (%)")
        ax1.legend(*scatter.legend_elements(), title="Ventilador (0=Off, 1=On)")
        ax1.grid(True, linestyle=':', alpha=0.6)

        # B. Loss Curve (Red Neuronal)
        ax2.plot(nn_model.loss_curve_, color='#673AB7', linewidth=2)
        ax2.set_title("Entrenamiento Red Neuronal (Loss Curve)")
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Error (Loss)")
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_confusion_matrices(self, parent_frame):
        """Gráficos: Matrices de Confusión"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)

        # A. Matriz Regresión Logística
        cm_log = confusion_matrix(y_test_log, y_pred_log)
        disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Apagado", "Encendido"])
        disp_log.plot(ax=ax1, cmap='Blues', colorbar=False)
        ax1.set_title("Matriz: Control Ambiental\n(Reg. Logística)")

        # B. Matriz Red Neuronal
        cm_nn = confusion_matrix(y_test_nn, y_pred_nn)
        disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=["Normal", "Dosificar"])
        disp_nn.plot(ax=ax2, cmap='Purples', colorbar=False)
        ax2.set_title("Matriz: Control Químico\n(Red Neuronal)")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except:
        pass

    app = SmartPotApp(root)
    root.mainloop()