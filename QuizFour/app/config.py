class Config:
    """
    Configuración centralizada y Base de Conocimiento para los cultivos.
    """
    APP_NAME = "SmartPot AI Core"
    VERSION = "1.0.0"

    # Base de conocimiento: Parámetros ideales
    CULTIVOS = {
        'tomate': {
            'luz_min': 300.0,
            'humedad': {'min': 60.0, 'max': 80.0},
            'ph': {'min': 5.5, 'max': 6.5},
            'tds': {'min': 1400.0, 'max': 3500.0},
            'temp': {'min': 20.0, 'max': 28.0}
        },
        'lechuga': {
            'luz_min': 200.0,
            'humedad': {'min': 50.0, 'max': 70.0},
            'ph': {'min': 5.5, 'max': 6.5},
            'tds': {'min': 560.0, 'max': 840.0},
            'temp': {'min': 15.0, 'max': 22.0}
        }
    }

    @staticmethod
    def get_rules(cultivo_nombre):
        return Config.CULTIVOS.get(cultivo_nombre.lower())