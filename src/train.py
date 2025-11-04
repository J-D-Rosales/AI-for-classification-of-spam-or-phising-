# src/train.py

"""
Script para entrenar el modelo de detección de spam/phishing.

Pasos:
1. Cargar todos los correos (data_loader.load_all_emails).
2. Dividir en train/test.
3. Vectorizar el texto con TF-IDF.
4. Entrenar un modelo de regresión logística.
5. Evaluar y mostrar métricas.
6. Guardar el vectorizador y el modelo en la carpeta 'models/'.
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from .data_loader import load_all_emails
from .preprocess import create_tfidf_vectorizer

# Carpeta donde guardaremos los modelos entrenados
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_model():
    # 1. Cargar datos
    data = load_all_emails()

    # X = textos, y = etiquetas (0/1)
    X = data["text"]
    y = data["label"]

    # 2. Dividir en train y test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # mantiene la proporción spam/no spam en train y test
    )

    # 3. Crear y ajustar el vectorizador TF-IDF
    vectorizer = create_tfidf_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Definir el modelo (Regresión Logística)
    model = LogisticRegression(
        max_iter=1000,  # más iteraciones para asegurar convergencia
        n_jobs=-1       # usa todos los núcleos disponibles
    )

    # Entrenar el modelo
    model.fit(X_train_vec, y_train)

    # 5. Evaluar el modelo
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]  # probabilidad de clase 1 (spam)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC: {auc:.4f}")
    except Exception as e:
        print("No se pudo calcular AUC:", e)

    # 6. Guardar vectorizador y modelo
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(model, MODELS_DIR / "classifier.pkl")

    print("Modelos guardados en la carpeta 'models/'.")


if __name__ == "__main__":
    train_model()
