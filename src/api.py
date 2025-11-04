# src/api.py

"""
API sencilla con FastAPI para exponer el detector de spam como servicio web.

Endpoint principal:
/predict  (POST)

Body de ejemplo:
{
    "text": "contenido del correo..."
}

Respuesta:
{
    "is_spam": true,
    "spam_probability": 0.9732
}
"""

from fastapi import FastAPI
from pydantic import BaseModel

from .predict import SpamDetector

# Creamos la app de FastAPI
app = FastAPI(
    title="Phishing Email Detector API",
    description="API para detectar si un email es spam/phishing o legítimo.",
    version="1.0.0",
)

# Cargamos el detector una sola vez al iniciar el servidor
detector = SpamDetector()


class EmailInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    is_spam: bool
    spam_probability: float


@app.post("/predict", response_model=PredictionOutput)
def predict_spam(email: EmailInput):
    """
    Recibe el texto de un correo y devuelve si es spam y con qué probabilidad.
    """
    result = detector.predict(email.text)
    return PredictionOutput(**result)
