from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def train_models(df):
    """
    Recibe un DataFrame, entrena los modelos y retorna un diccionario
    con los objetos entrenados y las métricas.
    """
    # --- MODELO A: Regresión Logística (Ventilador) ---
    X_log = df[['DHT_temp', 'DHT_humidity']]
    y_log = df['ex_fan']
    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_log, y_train_log)
    y_pred_log = log_reg.predict(X_test_log)
    acc_log = accuracy_score(y_test_log, y_pred_log)

    # --- MODELO B: Red Neuronal (pH) ---
    X_nn = df[['pH', 'TDS']]
    y_nn = df['pH_reducer']
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_nn_scaled = scaler.fit_transform(X_train_nn)
    X_test_nn_scaled = scaler.transform(X_test_nn)

    nn_model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=2000, random_state=42)
    nn_model.fit(X_train_nn_scaled, y_train_nn)
    y_pred_nn = nn_model.predict(X_test_nn_scaled)
    acc_nn = accuracy_score(y_test_nn, y_pred_nn)

    # Retornamos todo lo necesario en un paquete (diccionario)
    # También retornamos y_test/y_pred por si queremos graficar matrices de confusión
    return {
        'log_reg': log_reg,
        'nn_model': nn_model,
        'scaler': scaler,
        'metrics': {'log_acc': acc_log, 'nn_acc': acc_nn},
        'test_data': {
            'y_test_log': y_test_log, 'y_pred_log': y_pred_log,
            'y_test_nn': y_test_nn, 'y_pred_nn': y_pred_nn
        }
    }
