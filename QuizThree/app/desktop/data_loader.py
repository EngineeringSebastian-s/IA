import numpy as np
import pandas as pd


def get_data():
    """
    Genera los datos sintéticos o carga un CSV.
    Retorna un DataFrame de Pandas.
    """
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

    df = pd.DataFrame(synthetic_data, columns=['pH', 'TDS', 'DHT_temp', 'DHT_humidity', 'pH_reducer', 'ex_fan'])
    return df
