import time
from app.config import Config
from app.models import LecturaSensor
from app.agent import ReactiveAgent
from app.actuators import ActuatorSystem

# Datos simulados (los que me pasaste)
DATOS_ENTRADA = [
    {'id': 1, 'Atmósfera': 24, 'Brillo': 0, 'Humedad': 40, 'PH': 0, 'TDS': 0, 'Temperatura': 24, 'Fecha': '2025...'},
    {'id': 2, 'Atmósfera': 24, 'Brillo': 140.6, 'Humedad': 40, 'PH': 2.07, 'TDS': 225.8, 'Temperatura': 24,
     'Fecha': '2025...'},
    {'id': 3, 'Atmósfera': 33.1, 'Brillo': 140.6, 'Humedad': 45, 'PH': 2.07, 'TDS': 225.8, 'Temperatura': 33.1,
     'Fecha': '2025...'},
    {'id': 4, 'Atmósfera': 53.3, 'Brillo': 3.9, 'Humedad': 80.5, 'PH': 1.03, 'TDS': 70.3, 'Temperatura': 53.3,
     'Fecha': '2025...'},
    {'id': 5, 'Atmósfera': 22.5, 'Brillo': 248.3, 'Humedad': 80.5, 'PH': 2.17, 'TDS': 240.5, 'Temperatura': 22.5,
     'Fecha': '2025...'}
]


def run():
    print(f"--- INICIANDO {Config.APP_NAME} ---")

    # 1. Configuración Inicial
    tipo_cultivo = "tomate"
    reglas = Config.get_rules(tipo_cultivo)

    # 2. Instanciar Agente y Actuadores
    cerebro = ReactiveAgent(reglas)
    hardware = ActuatorSystem()

    print(f"🌱 Cultivo seleccionado: {tipo_cultivo.upper()}")

    # 3. Bucle Principal (Simulación de lectura continua)
    for raw_data in DATOS_ENTRADA:
        # A. Percepción (Convertir dict a objeto)
        sensor_data = LecturaSensor.from_dict(raw_data)

        print(
            f"\n📡 Leyendo Sensores [ID: {sensor_data.id_registro}] | Temp: {sensor_data.temperatura}°C | pH: {sensor_data.ph}")

        # B. Razonamiento (Agente decide)
        acciones_recomendadas = cerebro.percibir_y_reaccionar(sensor_data)

        # C. Actuación (Ejecutar órdenes)
        hardware.ejecutar(acciones_recomendadas)

        # Simular delay entre lecturas
        time.sleep(1)


if __name__ == "__main__":
    run()