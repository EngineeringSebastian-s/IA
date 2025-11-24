import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class SmartPotApp:
    def __init__(self, root, df, brain):
        """
        :param root: Ventana principal de Tkinter
        :param df: DataFrame con los datos (para gráficas)
        :param brain: Diccionario con modelos entrenados y scaler
        """
        self.root = root
        self.df = df
        self.brain = brain  # Aquí están los modelos

        self.root.title("SmartPot - Dashboard de Control y Diagnóstico")
        self.root.geometry("1200x700")

        # Estilos visuales
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 11))

        self.setup_ui()

    def setup_ui(self):
        # --- LAYOUT PRINCIPAL ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # COLUMNA IZQUIERDA
        control_panel = ttk.LabelFrame(main_frame, text=" Simulador de Sensores ", padding=15)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.create_slider(control_panel, "🌡️ Temperatura (°C)", 10, 40, 25.5, 0.1, 0, 'scale_temp')
        self.create_slider(control_panel, "💧 Humedad (%)", 0, 100, 60, 1, 2, 'scale_hum')
        ttk.Separator(control_panel, orient='horizontal').grid(row=4, column=0, sticky="ew", pady=15)
        self.create_slider(control_panel, "⚗️ pH del Agua", 0, 14, 7.0, 0.01, 5, 'scale_ph')
        self.create_slider(control_panel, "🧂 TDS (ppm)", 0, 1000, 500, 10, 7, 'scale_tds')

        btn_predict = ttk.Button(control_panel, text="ANALIZAR CONDICIONES", command=self.predict_actions)
        btn_predict.grid(row=9, column=0, pady=25, sticky="ew")

        self.res_frame = ttk.LabelFrame(control_panel, text=" Estado del Sistema (IA) ", padding=15)
        self.res_frame.grid(row=10, column=0, sticky="ew")

        self.lbl_fan = ttk.Label(self.res_frame, text="Ventilador: ---", style="Result.TLabel", foreground="gray")
        self.lbl_fan.pack(anchor="w", pady=5)
        self.lbl_ph = ttk.Label(self.res_frame, text="Reductor pH: ---", style="Result.TLabel", foreground="gray")
        self.lbl_ph.pack(anchor="w", pady=5)

        # COLUMNA DERECHA
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_behavior = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_behavior, text="Dinámica del Modelo")

        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="Evaluación de Precisión")

        self.plot_behavior_charts(self.tab_behavior)
        self.plot_confusion_matrices(self.tab_metrics)

    def create_slider(self, parent, text, vmin, vmax, vdef, res, row, attr_name):
        ttk.Label(parent, text=text, style="Header.TLabel").grid(row=row, column=0, sticky="w", pady=(5, 0))
        scale = tk.Scale(parent, from_=vmin, to=vmax, orient=tk.HORIZONTAL, resolution=res, length=240,
                         activebackground="#2196F3", highlightthickness=0)
        scale.set(vdef)
        scale.grid(row=row + 1, column=0, pady=(0, 10))
        setattr(self, attr_name, scale)

    def predict_actions(self):
        try:
            t = self.scale_temp.get()
            h = self.scale_hum.get()
            ph = self.scale_ph.get()
            tds = self.scale_tds.get()

            # DataFrame inputs
            in_log = pd.DataFrame([[t, h]], columns=['DHT_temp', 'DHT_humidity'])
            in_nn = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

            # Usamos los modelos almacenados en self.brain
            log_reg = self.brain['log_reg']
            nn_model = self.brain['nn_model']
            scaler = self.brain['scaler']

            fan_act = log_reg.predict(in_log)[0]
            fan_prob = log_reg.predict_proba(in_log)[0][1]

            in_nn_scaled = scaler.transform(in_nn)
            ph_act = nn_model.predict(in_nn_scaled)[0]

            if fan_act == 1:
                self.lbl_fan.config(text=f"💨 Ventilador: ACTIVADO ({fan_prob * 100:.1f}%)", foreground="red")
            else:
                self.lbl_fan.config(text=f"💨 Ventilador: APAGADO ({fan_prob * 100:.1f}%)", foreground="green")

            if ph_act == 1:
                self.lbl_ph.config(text="🧪 Reductor pH: DOSIFICAR", foreground="red")
            else:
                self.lbl_ph.config(text="🧪 Reductor pH: ESTABLE", foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo: {e}")

    def plot_behavior_charts(self, parent_frame):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=100)

        # Usamos self.df (datos cargados)
        scatter = ax1.scatter(self.df['DHT_temp'], self.df['DHT_humidity'],
                              c=self.df['ex_fan'], cmap='coolwarm', alpha=0.6, edgecolors='w')
        ax1.set_title("Distribución: Temperatura vs Humedad")
        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Humedad (%)")
        ax1.legend(*scatter.legend_elements(), title="Ventilador")
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Usamos el modelo de la red para la curva de pérdida
        nn_model = self.brain['nn_model']
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=100)

        # Recuperamos datos de test almacenados en brain['test_data']
        td = self.brain['test_data']

        cm_log = confusion_matrix(td['y_test_log'], td['y_pred_log'])
        disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Apagado", "Encendido"])
        disp_log.plot(ax=ax1, cmap='Blues', colorbar=False)
        ax1.set_title(f"Reg. Logística (Acc: {self.brain['metrics']['log_acc']:.2f})")

        cm_nn = confusion_matrix(td['y_test_nn'], td['y_pred_nn'])
        disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=["Normal", "Dosificar"])
        disp_nn.plot(ax=ax2, cmap='Purples', colorbar=False)
        ax2.set_title(f"Red Neuronal (Acc: {self.brain['metrics']['nn_acc']:.2f})")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
