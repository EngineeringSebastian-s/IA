import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Importamos nuestros módulos locales
from data_loader import get_data
from model_trainer import train_models
from utils import apply_custom_styles

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SmartPot Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS
apply_custom_styles()

# --- CARGA Y ENTRENAMIENTO ---
df_synth = get_data()
brain = train_models(df_synth)

# Desempaquetamos los modelos para usarlos fácil
log_reg = brain['log_reg']
nn_model = brain['nn_model']
scaler = brain['scaler']
metrics = brain['metrics']

# --- INTERFAZ GRÁFICA ---
st.title("🌱 SmartPot - Dashboard de Control IoT")
st.markdown("Sistema inteligente basado en **Regresión Logística** y **Redes Neuronales**.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.info("Ajusta los sensores para ver la reacción de la IA.")

    st.subheader("Variables Ambientales")
    temp = st.slider("🌡️ Temperatura (°C)", 10.0, 40.0, 25.5, 0.1)
    hum = st.slider("💧 Humedad (%)", 30, 95, 60, 1)
    st.markdown("---")
    st.subheader("Variables del Agua")
    ph = st.slider("⚗️ pH del Agua", 4.0, 9.0, 7.0, 0.01)
    tds = st.slider("🧂 TDS (ppm)", 200, 900, 500, 10)

    st.caption("Modelo entrenado con 500 muestras sintéticas.")

# --- LÓGICA DE PREDICCIÓN ---
in_log = pd.DataFrame([[temp, hum]], columns=['DHT_temp', 'DHT_humidity'])
in_nn = pd.DataFrame([[ph, tds]], columns=['pH', 'TDS'])

fan_act = log_reg.predict(in_log)[0]
fan_prob = log_reg.predict_proba(in_log)[0][1]

in_nn_scaled = scaler.transform(in_nn)
ph_act = nn_model.predict(in_nn_scaled)[0]

# --- PANEL DE RESULTADOS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("💨 Control Ambiental (Fan)")
    if fan_act == 1:
        st.error(f"**VENTILADOR ENCENDIDO**\n\nProbabilidad: {fan_prob * 100:.1f}%")
    else:
        st.success(f"**VENTILADOR APAGADO**\n\nProbabilidad: {fan_prob * 100:.1f}%")

with col2:
    st.subheader("🧪 Control Químico (pH)")
    if ph_act == 1:
        st.error("**DOSIFICAR REDUCTOR**\n\nEl pH está alto.")
    else:
        st.success("**NIVEL ESTABLE**\n\nNo se requieren químicos.")

st.markdown("---")

# --- PESTAÑAS DE GRÁFICOS ---
tab1, tab2 = st.tabs(["📊 Dinámica del Modelo", "📈 Evaluación de Precisión"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Distribución Temp vs Humedad")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        # Usamos df_synth importado
        ax1.scatter(df_synth['DHT_temp'], df_synth['DHT_humidity'],
                    c=df_synth['ex_fan'], cmap='coolwarm', alpha=0.5, edgecolors='none')
        ax1.scatter([temp], [hum], color='lime', s=200, marker='*', edgecolors='black', label='Tu Selección')
        ax1.set_xlabel("Temperatura")
        ax1.set_ylabel("Humedad")
        ax1.legend()
        st.pyplot(fig1)

    with col_g2:
        st.markdown("#### Curva de Aprendizaje (Red Neuronal)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(nn_model.loss_curve_, color='#673AB7', linewidth=2)
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Loss")
        st.pyplot(fig2)

with tab2:
    col_m1, col_m2 = st.columns(2)
    td = brain['test_data']  # Datos de prueba guardados

    with col_m1:
        st.markdown(f"#### Matriz Regresión Logística (Acc: {metrics['log_acc']:.2f})")
        fig3, ax3 = plt.subplots()
        cm_log = confusion_matrix(td['y_test_log'], td['y_pred_log'])
        disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Apagado", "Encendido"])
        disp_log.plot(ax=ax3, cmap='Blues', colorbar=False)
        st.pyplot(fig3)

    with col_m2:
        st.markdown(f"#### Matriz Red Neuronal (Acc: {metrics['nn_acc']:.2f})")
        fig4, ax4 = plt.subplots()
        cm_nn = confusion_matrix(td['y_test_nn'], td['y_pred_nn'])
        disp_nn = ConfusionMatrixDisplay(confusion_matrix=cm_nn, display_labels=["Normal", "Dosificar"])
        disp_nn.plot(ax=ax4, cmap='Purples', colorbar=False)
        st.pyplot(fig4)