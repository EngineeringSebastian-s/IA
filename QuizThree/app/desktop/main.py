import tkinter as tk

from data_loader import get_data
from gui_app import SmartPotApp
from model_trainer import train_models


def main():
    print("1. Cargando datos...")
    df = get_data()
    print(f"   -> Datos cargados: {len(df)} registros.")

    print("2. Entrenando modelos de IA...")
    # 'brain' es un diccionario con {log_reg, nn_model, scaler, metrics...}
    brain = train_models(df)
    print(f"   -> Precisión Logística: {brain['metrics']['log_acc']:.2f}")
    print(f"   -> Precisión Red Neuronal: {brain['metrics']['nn_acc']:.2f}")

    print("3. Iniciando Interfaz Gráfica...")
    root = tk.Tk()

    # Intentar cargar tema si existe
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except:
        pass

    # Pasamos 'df' y 'brain' a la APP para que pueda usar los modelos y gráficas
    app = SmartPotApp(root, df, brain)
    root.mainloop()


if __name__ == "__main__":
    main()
