import streamlit as st
import pandas as pd
import time
from datetime import datetime

# --- IMPORTAMOS TU LÓGICA EXISTENTE ---
# Asegúrate de ejecutar streamlit desde la carpeta raíz del proyecto
try:
    from app.config import Config
    from app.models import LecturaSensor
    from app.agent import ReactiveAgent
except ImportError:
    st.error(
        "⚠️ Error de importación: Asegúrate de correr esto desde la raíz del proyecto con 'python -m streamlit run app/interface.py'")
    st.stop()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SmartPot Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS (Adaptado de tu ejemplo) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .action-alert {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .status-ok {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .status-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. SIDEBAR: CONFIGURACIÓN Y SENSORES
# ==========================================
with st.sidebar:
    st.title("🎛️ Panel de Control")

    # Selector de Cultivo (Carga reglas desde config.py)
    cultivos_disponibles = list(Config.CULTIVOS.keys())
    seleccion_cultivo = st.selectbox("🌱 Seleccionar Cultivo", cultivos_disponibles, index=0)

    # Cargamos las reglas específicas para mostrar límites en los sliders si quisiéramos
    reglas = Config.get_rules(seleccion_cultivo)

    st.markdown("---")
    st.subheader("Simulación de Sensores")

    # Sliders para modificar los valores de entrada (Input Data)
    # Los valores por defecto están puestos para estar 'cerca' de lo ideal y facilitar pruebas
    temp = st.slider("🌡️ Temperatura (°C)", 0.0, 50.0, 24.0, 0.1)
    hum = st.slider("💧 Humedad (%)", 0, 100, 65, 1)
    luz = st.slider("☀️ Brillo / Luz (Lux)", 0, 1000, 350, 10)

    st.markdown("---")
    ph = st.slider("⚗️ pH del Agua", 0.0, 14.0, 6.0, 0.1)
    tds = st.slider("🧂 TDS (ppm)", 0, 5000, 1500, 50)
    atmosfera = st.slider("☁️ Atmósfera (CO2/Voc)", 0, 100, 24, 1)

    # Botón para forzar actualización (aunque Streamlit es reactivo por defecto)
    st.caption(f"Configuración activa: {seleccion_cultivo.upper()}")

# ==========================================
# 2. LÓGICA DEL AGENTE (BACKEND)
# ==========================================

# 1. Crear objeto de datos (Models)
datos_sensor = LecturaSensor(
    id_registro=999,  # ID simulado
    atmosfera=atmosfera,
    brillo=float(luz),
    humedad=float(hum),
    ph=ph,
    tds=float(tds),
    temperatura=temp,
    fecha=datetime.now().isoformat()
)

# 2. Instanciar el Agente (Agent)
try:
    cerebro = ReactiveAgent(reglas)
    # 3. El Agente 'Piensa'
    acciones_recomendadas = cerebro.percibir_y_reaccionar(datos_sensor)
except Exception as e:
    st.error(f"Error en el agente: {e}")
    acciones_recomendadas = []

# ==========================================
# 3. INTERFAZ PRINCIPAL (FRONTEND)
# ==========================================

st.title(f"SmartPot AI: Monitor de {seleccion_cultivo.capitalize()}")
st.markdown("Sistema de **Agente Reactivo** en tiempo real. Modifica los sensores a la izquierda para ver la reacción.")

# --- METRIC CARDS (Estado Actual) ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Temperatura", f"{temp} °C", delta_color="off")
col2.metric("Humedad", f"{hum} %", delta_color="off")
col3.metric("Luz", f"{luz} Lux", delta_color="off")
col4.metric("pH", f"{ph}", delta_color="off")
col5.metric("TDS", f"{tds} ppm", delta_color="off")

st.markdown("---")

# --- ÁREA DE RESPUESTA DEL AGENTE ---
c_resumen, c_acciones = st.columns([1, 2])

with c_resumen:
    st.subheader("📊 Estado del Cultivo")

    # Lógica simple para determinar estado general visual
    if not acciones_recomendadas:
        st.success("✅ **ÓPTIMO**\n\nTodos los parámetros están dentro del rango ideal.")
        estado_general = "ok"
    else:
        st.warning(f"⚠️ **ATENCIÓN REQUERIDA**\n\nSe han detectado {len(acciones_recomendadas)} anomalía(s).")
        estado_general = "alert"

    # Mostrar rangos ideales de referencia (Viene de config.py)
    with st.expander("Ver Rangos Ideales"):
        st.json(reglas)

with c_acciones:
    st.subheader("⚡ Acciones del Agente (Actuators)")

    if not acciones_recomendadas:
        st.markdown('<div class="action-alert status-ok">Sistema en Standby. Sin acciones necesarias.</div>',
                    unsafe_allow_html=True)
    else:
        for accion in acciones_recomendadas:
            # Parseamos un poco el texto para darle estilo según el tipo de acción
            estilo = "status-warning"
            icono = "🔧"

            if "CRÍTICO" in accion.upper() or "APAGAR" in accion.upper():
                estilo = "status-danger"
                icono = "🚨"
            elif "ENCENDER" in accion.upper() or "ACTIVAR" in accion.upper():
                estilo = "status-warning"
                icono = "⚡"
            elif "DOSIFICAR" in accion.upper():
                estilo = "status-warning"
                icono = "🧪"

            st.markdown(f'<div class="action-alert {estilo}">{icono} {accion}</div>', unsafe_allow_html=True)

# --- VISUALIZACIÓN GRÁFICA (Comparativa) ---
st.markdown("### 📈 Visualización de Parámetros vs Reglas")

tab_clima, tab_agua = st.tabs(["🌡️ Clima", "💧 Riego y Nutrientes"])

with tab_clima:
    col_g1, col_g2 = st.columns(2)

    # Gráfico de barras simple usando Streamlit native o Altair/Matplotlib
    # Aquí hacemos algo rápido visual con barras de progreso custom

    with col_g1:
        st.write("**Temperatura**")
        min_t, max_t = reglas['temp']['min'], reglas['temp']['max']
        # Calculamos porcentaje relativo para la barra (visualización simple)
        st.write(f"Rango Ideal: {min_t} - {max_t} °C")
        if temp < min_t:
            st.progress(0.2); st.caption("🥶 Muy Frío")
        elif temp > max_t:
            st.progress(1.0); st.caption("🔥 Muy Caliente")
        else:
            st.progress(0.5); st.caption("✅ Ideal")

    with col_g2:
        st.write("**Humedad**")
        min_h, max_h = reglas['humedad']['min'], reglas['humedad']['max']
        st.write(f"Rango Ideal: {min_h} - {max_h} %")
        if hum < min_h:
            st.progress(0.2); st.caption("🌵 Seco")
        elif hum > max_h:
            st.progress(1.0); st.caption("💧 Excesivo")
        else:
            st.progress(0.5); st.caption("✅ Ideal")

with tab_agua:
    # Usamos un dataframe para graficar pH
    chart_data = pd.DataFrame({
        'Parametro': ['Tu pH', 'Min Ideal', 'Max Ideal'],
        'Valor': [ph, reglas['ph']['min'], reglas['ph']['max']]
    })
    st.bar_chart(chart_data, x='Parametro', y='Valor',
                 color=["#FF0000" if (ph < reglas['ph']['min'] or ph > reglas['ph']['max']) else "#00FF00"])
    st.caption("Si la barra de 'Tu pH' está fuera de las otras dos, el agente actuará.")