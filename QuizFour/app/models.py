from dataclasses import dataclass
from datetime import datetime

@dataclass
class LecturaSensor:
    """
    Modelo de datos que representa una fila de tus registros.
    """
    id_registro: int
    atmosfera: float
    brillo: float
    humedad: float
    ph: float
    tds: float
    temperatura: float
    fecha: str  # Podría ser datetime, pero lo dejaremos str por simplicidad inicial

    @classmethod
    def from_dict(cls, data):
        """Helper para convertir un diccionario crudo en un objeto estructurado"""
        return cls(
            id_registro=int(data.get('id', 0)),
            atmosfera=float(data.get('Atmósfera', 0)),
            brillo=float(data.get('Brillo', 0)),
            humedad=float(data.get('Humedad', 0)),
            ph=float(data.get('PH', 0)),
            tds=float(data.get('TDS', 0)),
            temperatura=float(data.get('Temperatura', 0)),
            fecha=data.get('Fecha', '')
        )