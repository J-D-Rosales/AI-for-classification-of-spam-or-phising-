# src/predict.py

"""
Cargar el modelo entrenado y hacer predicciones a partir de un texto.

Pensado para correos en INGLÉS.

- Devuelve probabilidad de SPAM.
- Usa un UMBRAL (threshold) para decidir si es spam o no.
"""

from pathlib import Path
import joblib

# raíz del proyecto: .../clasificar_spam/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# carpeta de modelos: .../clasificar_spam/models/
MODELS_DIR = PROJECT_ROOT / "models"


class SpamDetector:
    def __init__(self, threshold: float = 0.7):
        """
        Carga el vectorizador TF-IDF y el modelo de clasificación desde disco.

        Parámetros:
        - threshold: umbral para decidir si es spam (0.0 - 1.0).
                     Si probabilidad_de_spam >= threshold -> is_spam = True
        """
        self.threshold = threshold

        vectorizer_path = MODELS_DIR / "vectorizer.pkl"
        model_path = MODELS_DIR / "classifier.pkl"

        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "No se encontraron los archivos 'vectorizer.pkl' y 'classifier.pkl'. "
                "Primero ejecuta: python -m src.train"
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        """
        Recibe un solo string (contenido del correo EN INGLÉS) y devuelve:
        - is_spam: True si se predice spam/phishing, False si no.
        - spam_probability: probabilidad de que sea spam (entre 0 y 1).

        Usa el umbral self.threshold para decidir.
        """
        X_vec = self.vectorizer.transform([text])

        proba_spam = self.model.predict_proba(X_vec)[0, 1]

        is_spam = bool(proba_spam >= self.threshold)

        return {
            "is_spam": is_spam,
            "spam_probability": float(proba_spam),
            "threshold": float(self.threshold),
        }


if __name__ == "__main__":
    detector = SpamDetector(threshold=0.7)
    ejemplo = "Professor, I am sending you my assignment by email."
    resultado = detector.predict(ejemplo)
    print("Texto:", ejemplo)
    print("Resultado:", resultado)
