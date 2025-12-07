# Portafolio de Ingeniería de Inteligencia Artificial

Este repositorio consolida los proyectos prácticos desarrollados para la asignatura de **Inteligencia Artificial**. Abarca desde la IA Simbólica clásica hasta algoritmos de Machine Learning y Agentes Inteligentes aplicados a problemas de ingeniería.

**Autor:** Sebastián López Osorno
**Institución:** Politécnico Colombiano Jaime Isaza Cadavid

-----

## 📂 Estructura del Portafolio

El proyecto se divide en 4 módulos (Quizzes), cada uno enfocado en un paradigma distinto de la IA:

### 1\. Inteligencia Artificial Simbólica (Lógica y Conocimiento)

Ubicación: `/QuizOne`

  * **🕯️ Sistema Experto (Expert System):**

      * **Problema:** Identificación de patrones de velas japonesas ("Envolventes") en mercados financieros.
      * **Tecnología:** CLIPS (Motor de inferencia), Python (`clipspy`), Tkinter.
      * **Funcionamiento:** Utiliza reglas lógicas estrictas para determinar tendencias alcistas o bajistas.

  * **☁️ Lógica Difusa (Fuzzy Logic):**

      * **Problema:** Toma de decisiones de inversión bajo incertidumbre.
      * **Tecnología:** `scikit-fuzzy`.
      * **Funcionamiento:** Mapea variables de entrada (tamaño de vela, nivel de envolvimiento) a conjuntos difusos para recomendar una intensidad de inversión (Poco, Normal, Mucho).

### 2\. Computación Evolutiva (Algoritmos Genéticos)

Ubicación: `/QuizTwo`

  * **🧬 Coloreado de Imágenes (Image Colorization):**
      * **Objetivo:** Reconstruir la información de color (RGB) de una imagen en escala de grises mediante evolución estocástica.
      * **Método:** Población de matrices de píxeles que mutan y se cruzan para minimizar el error de luminancia y crominancia.
  * **🪂 Física del Paracaidista:**
      * **Objetivo:** Optimizar los parámetros de salto (altura de apertura, ángulo, corrección de viento) para aterrizar en coordenadas específicas $(0,0)$ con velocidad segura.

### 3\. Machine Learning (Aprendizaje Supervisado)

Ubicación: `/QuizThree`

  * **🌱 SmartPot AI (Clasificación):**
      * **Contexto:** Sistema IoT para cultivos hidropónicos.
      * **Modelos Comparados:**
        1.  **Regresión Logística:** Para control ambiental (Ventiladores).
        2.  **Red Neuronal (MLP):** Para control químico (Dosificación de pH).
      * **Interfaces:** Disponible en versión Escritorio (Tkinter) y Web (Streamlit).

### 4\. Agentes Inteligentes

Ubicación: `/QuizFour`

  * **🤖 Agente Reactivo Simple:**
      * **Contexto:** Automatización de SmartPot.
      * **Arquitectura:** Percepción $\rightarrow$ Reglas (Base de Conocimiento) $\rightarrow$ Acción.
      * **Simulación:** Dashboard interactivo que demuestra cómo el agente reacciona en tiempo real a cambios en sensores simulados (pH, Humedad, Luz).

-----

## 🛠️ Tecnologías y Librerías

El proyecto está construido íntegramente en **Python**. Las principales dependencias son:

| Categoría | Librerías |
| :--- | :--- |
| **Ciencia de Datos** | `numpy`, `pandas`, `matplotlib`, `scipy` |
| **IA & ML** | `scikit-learn`, `scikit-fuzzy`, `clipspy` (CLIPS wrapper) |
| **Interfaces (GUI)** | `tkinter` (Standard Lib), `streamlit` |
| **Procesamiento Imag.** | `pillow` (PIL) |

-----

## 🚀 Instalación y Ejecución General

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/tu-usuario/engineeringsebastian-s-ia.git
    cd engineeringsebastian-s-ia
    ```

2.  **Crear entorno virtual (Recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**
    Puedes instalar las dependencias globales ubicadas en la raíz:

    ```bash
    pip install -r requirements.txt
    ```

    *Nota: Cada subproyecto puede tener requerimientos específicos adicionales en su propia carpeta.*

### Guía de Ejecución por Módulo

#### 🔹 Quiz 1: Sistema Experto de Velas

```bash
python QuizOne/ExpertSystem/src/main.py
```

#### 🔹 Quiz 1: Lógica Difusa de Inversión

```bash
python QuizOne/FuzzyLogic/src/main.py
```

#### 🔹 Quiz 2: Algoritmo Genético (Imágenes)

```bash
# Asegúrate de configurar la ruta de la imagen en el script main.py si es necesario
python QuizTwo/ImageColorizationGA/main.py
```

#### 🔹 Quiz 3: Modelos de ML (SmartPot)

  * **Versión Web (Dashboard completo):**
    ```bash
    streamlit run QuizThree/app/web.py
    ```
  * **Versión Escritorio:**
    ```bash
    python QuizThree/app/desktop.py
    ```

#### 🔹 Quiz 4: Agente Inteligente

```bash
streamlit run QuizFour/app/interface.py
```

-----

## 📄 Licencia

Este proyecto tiene fines académicos y educativos como parte del programa de Ingeniería.

-----
