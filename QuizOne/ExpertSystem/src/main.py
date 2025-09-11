import tkinter as tk
from tkinter import messagebox
import clips


def identificar_vela_envolvente(precios_vela1, precios_vela2):
    """
    Identifica si dos velas forman un patrón envolvente alcista o bajista
    utilizando un sistema de reglas con clipsy.
    """
    env = clips.Environment()
    
    deftemplate_vela = """
    (deftemplate vela (slot id (type INTEGER)) (slot apertura (type FLOAT)) (slot cierre (type FLOAT)))
    """
    env.build(deftemplate_vela)

    deftemplate_patron = """
    (deftemplate patron (slot tipo (type STRING)))
    """
    env.build(deftemplate_patron)

    defrule_bajista = """
    (defrule identificar-envolvente-bajista
        (vela (id 1) (apertura ?a1) (cierre ?c1))
        (vela (id 2) (apertura ?a2) (cierre ?c2))
        (test (and (> ?c1 ?a1) (< ?c2 ?a2) (>= ?a2 ?c1) (< ?c2 ?a1)))
        =>
        (assert (patron (tipo "Envolvente Bajista"))))
    """
    env.build(defrule_bajista)

    defrule_alcista = """
    (defrule identificar-envolvente-alcista
        (vela (id 1) (apertura ?a1) (cierre ?c1))
        (vela (id 2) (apertura ?a2) (cierre ?c2))
        (test (and (< ?c1 ?a1) (> ?c2 ?a2) (<= ?a2 ?c1) (> ?c2 ?a1)))
        =>
        (assert (patron (tipo "Envolvente Alcista"))))
    """
    env.build(defrule_alcista)

    plantilla_vela = env.find_template('vela')
    plantilla_vela.assert_fact(id=1, apertura=float(precios_vela1[0]), cierre=float(precios_vela1[1]))
    plantilla_vela.assert_fact(id=2, apertura=float(precios_vela2[0]), cierre=float(precios_vela2[1]))

    env.run()

    patron_encontrado = None
    for fact in env.facts():
        if fact.template.name == 'patron':
            patron_encontrado = fact['tipo']
            break
            
    if patron_encontrado:
        return f"Se ha identificado un patrón de '{patron_encontrado}'."
    else:
        return "No se ha identificado un patrón envolvente."

# ==============================================================================
# CÓDIGO DE LA INTERFAZ GRÁFICA (GUI) CON TKINTER
# ==============================================================================
class AnalizadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Velas Envolventes")
        self.root.geometry("500x500") # Ancho x Alto
        self.root.resizable(False, False)

        # --- Frame para la entrada de datos ---
        frame_datos = tk.Frame(root, padx=10, pady=10)
        frame_datos.pack(fill="x", padx=10, pady=5)

        # --- Entradas para la Vela 1 ---
        tk.Label(frame_datos, text="Vela 1 (Anterior)", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=5)
        tk.Label(frame_datos, text="Precio Apertura:").grid(row=1, column=0, sticky="w")
        self.vela1_apertura = tk.Entry(frame_datos)
        self.vela1_apertura.grid(row=1, column=1)

        tk.Label(frame_datos, text="Precio Cierre:").grid(row=2, column=0, sticky="w")
        self.vela1_cierre = tk.Entry(frame_datos)
        self.vela1_cierre.grid(row=2, column=1)

        # --- Entradas para la Vela 2 ---
        tk.Label(frame_datos, text="Vela 2 (Actual)", font=("Arial", 10, "bold")).grid(row=0, column=2, columnspan=2, padx=20, pady=5)
        tk.Label(frame_datos, text="Precio Apertura:").grid(row=1, column=2, sticky="w", padx=(20,0))
        self.vela2_apertura = tk.Entry(frame_datos)
        self.vela2_apertura.grid(row=1, column=3)
        
        tk.Label(frame_datos, text="Precio Cierre:").grid(row=2, column=2, sticky="w", padx=(20,0))
        self.vela2_cierre = tk.Entry(frame_datos)
        self.vela2_cierre.grid(row=2, column=3)

        # --- Botón de Análisis ---
        self.boton_analizar = tk.Button(root, text="Analizar Patrón", command=self.analizar)
        self.boton_analizar.pack(pady=10)

        # --- Canvas para dibujar las velas ---
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white', relief='sunken', borderwidth=1)
        self.canvas.pack(pady=10)

        # --- Etiqueta para mostrar el resultado ---
        self.resultado_label = tk.Label(root, text="Ingrese los precios y presione 'Analizar'", font=("Arial", 12, "italic"))
        self.resultado_label.pack(pady=10)

    def dibujar_velas(self, p1, p2):
        self.canvas.delete("all") # Limpiar el canvas
        
        v1_open, v1_close = p1
        v2_open, v2_close = p2

        # Determinar el rango de precios para escalar el dibujo
        min_price = min(p1 + p2)
        max_price = max(p1 + p2)
        price_range = max_price - min_price
        if price_range == 0: price_range = 1 # Evitar división por cero

        # Función para escalar el precio a la altura del canvas
        def scale_price(price):
            return 190 - ((price - min_price) * 180 / price_range)

        # Dibujar Vela 1
        color_v1 = "green" if v1_close > v1_open else "red"
        self.canvas.create_rectangle(50, scale_price(v1_open), 90, scale_price(v1_close), fill=color_v1, outline="black")
        
        # Dibujar Vela 2
        color_v2 = "green" if v2_close > v2_open else "red"
        self.canvas.create_rectangle(110, scale_price(v2_open), 150, scale_price(v2_close), fill=color_v2, outline="black")


    def analizar(self):
        # 1. Obtener datos de los campos de entrada
        try:
            v1_open = float(self.vela1_apertura.get())
            v1_close = float(self.vela1_cierre.get())
            v2_open = float(self.vela2_apertura.get())
            v2_close = float(self.vela2_cierre.get())
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingrese solo números válidos en todos los campos.")
            return

        # 2. Preparar los datos
        vela1 = (v1_open, v1_close)
        vela2 = (v2_open, v2_close)

        # 3. Llamar a la función de análisis
        resultado = identificar_vela_envolvente(vela1, vela2)

        # 4. Actualizar la etiqueta de resultado
        self.resultado_label.config(text=resultado, font=("Arial", 12, "bold"))
        
        # 5. Dibujar las velas en el canvas
        self.dibujar_velas(vela1, vela2)


if __name__ == "__main__":
    ventana_principal = tk.Tk()
    app = AnalizadorApp(ventana_principal)
    ventana_principal.mainloop()
