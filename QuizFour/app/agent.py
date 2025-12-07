from app.models import LecturaSensor


class ReactiveAgent:
    def __init__(self, config_cultivo):
        self.rules = config_cultivo
        if not self.rules:
            raise ValueError("Configuración de cultivo no válida.")

    def percibir_y_reaccionar(self, sensor: LecturaSensor):
        """
        Ciclo de Inteligencia: Percepción -> Reglas -> Acción
        """
        acciones = []

        # 1. Evaluar LUZ
        if sensor.brillo < self.rules['luz_min']:
            acciones.append("ENCENDER_LUZ_UV")

        # 2. Evaluar TEMPERATURA
        if sensor.temperatura > self.rules['temp']['max']:
            acciones.append("ENCENDER_VENTILADORES")
            acciones.append("ENCENDER_ENFRIADOR_AGUA")
        elif sensor.temperatura < self.rules['temp']['min']:
            acciones.append("ENCENDER_CALEFACCION")

        # 3. Evaluar HUMEDAD
        if sensor.humedad < self.rules['humedad']['min']:
            acciones.append("ACTIVAR_BOMBA_RIEGO")
        elif sensor.humedad > self.rules['humedad']['max']:
            acciones.append("ACTIVAR_EXTRACTOR_AIRE")

        # 4. Evaluar PH (Crítico)
        if sensor.ph < self.rules['ph']['min']:
            acciones.append(f"DOSIFICAR_PH_UP (Nivel actual: {sensor.ph:.2f})")
        elif sensor.ph > self.rules['ph']['max']:
            acciones.append("DOSIFICAR_PH_DOWN")

        # 5. Evaluar TDS (Nutrientes)
        if sensor.tds < self.rules['tds']['min']:
            acciones.append("DOSIFICAR_NUTRIENTES_A_B")
        elif sensor.tds > self.rules['tds']['max']:
            acciones.append("DILUIR_SOLUCION_AGUA_PURA")

        return acciones