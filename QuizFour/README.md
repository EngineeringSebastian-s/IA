Aquí tienes una propuesta de `README.md` profesional, estructurado específicamente para desarrolladores y orientado a la implementación técnica del proyecto.

He sintetizado la información de tu documentación para que sea directa y funcional, añadiendo las secciones estándar de un repositorio de software (Instalación, Uso, Estructura).

-----

# 🌱 SmartPot: Agente Reactivo para Hidroponía Automatizada

**SmartPot** es un sistema de software diseñado para el control y monitoreo de jardines hidropónicos de precisión. Implementa un **Agente Inteligente Reactivo** capaz de percibir variables ambientales críticas (pH, temperatura, humedad, etc.) y ejecutar acciones correctivas inmediatas para mantener la homeostasis del cultivo.

El sistema simula un entorno IoT completo, permitiendo visualizar la lógica de decisión del agente a través de un Dashboard interactivo.

## 🏗 Arquitectura del Sistema

El proyecto se basa en el modelo de **Agente Reactivo Simple**. No mantiene un historial de estados pasados; su toma de decisiones es determinista y basada enteramente en la percepción actual (`Percept -> Action`).

[Image of simple reactive agent architecture diagram]

El flujo de datos sigue este ciclo:

1.  **Sensores (Input):** Capturan datos del entorno (simulados en la interfaz o leídos vía hardware).
2.  **Agente (Processing):** Compara los valores con la **Base de Conocimiento** (`config.py`).
3.  **Actuadores (Output):** Ejecutan órdenes (encender bombas, ventiladores, dosificadores) si los valores salen del rango ideal.

## 📂 Estructura del Proyecto

El código está modularizado para separar la lógica de negocio, la interfaz y la configuración. A continuación se describe el propósito de cada archivo dentro del paquete `app/`:

```text
/ (root)
├── app/
│   ├── __init__.py      # Inicializador del paquete
│   ├── actuators.py     # Sistema de actuación (simulación de hardware/GPIO)
│   ├── agent.py         # Lógica del Agente Reactivo (Cerebro)
│   ├── config.py        # Base de Conocimiento (Reglas y umbrales por cultivo)
│   ├── interface.py     # Frontend (Dashboard en Streamlit)
│   ├── main.py          # Punto de entrada para ejecución CLI (opcional)
│   └── models.py        # Definición de estructuras de datos (Dataclasses)
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación
```

### Descripción de Módulos Clave:

  * **`agent.py`**: Contiene la clase `ReactiveAgent`. Es el motor de inferencia que recibe un objeto `LecturaSensor` y devuelve una lista de acciones.
  * **`config.py`**: Diccionario centralizado que actúa como base de conocimiento. Aquí se definen los rangos `min` y `max` para pH, TDS, temperatura, etc., según el cultivo (Tomate, Lechuga).
  * **`interface.py`**: Implementación de la UI con **Streamlit**. Permite manipular los valores de entrada mediante sliders para probar la reactividad del agente en tiempo real.

## ⚙️ Requisitos Previos

  * **Python 3.8** o superior.
  * **Pip** (Gestor de paquetes de Python).

## 🚀 Instalación

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/tu-usuario/smartpot.git
    cd smartpot
    ```

2.  **Crear un entorno virtual (Recomendado):**

    ```bash
    # En Windows
    python -m venv venv
    venv\Scripts\activate

    # En Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    Asegúrate de tener el archivo `requirements.txt` en la raíz.

    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Ejecución

El proyecto cuenta con una interfaz gráfica web para facilitar la visualización y las pruebas del agente.

Para iniciar el Dashboard, ejecuta el siguiente comando desde la **raíz del proyecto**:

```bash
python -m streamlit run app/interface.py
```

Esto abrirá automáticamente una pestaña en tu navegador (usualmente en `http://localhost:8501`).

## 🛠 Configuración de Reglas

Para modificar los parámetros ideales de los cultivos o agregar nuevas especies, edita el archivo `app/config.py`.

**Ejemplo de estructura en `config.py`:**

```python
CULTIVOS = {
    'tomate': {
        'luz_min': 300.0,
        'humedad': {'min': 60.0, 'max': 80.0},
        'ph': {'min': 5.5, 'max': 6.5},
        # ... otros parámetros
    },
    'nuevo_cultivo': {
        # ... definir reglas aquí
    }
}
```

El agente leerá automáticamente esta configuración al reiniciarse.

## ✒️ Autor

**Sebastián López Osorno**

  * Politécnico Colombiano Jaime Isaza Cadavid
  * Contacto: sebastian\_lopez82221@elpoli.edu.co

-----

*Proyecto desarrollado como parte de la investigación sobre Agentes Inteligentes en Agricultura de Precisión.*