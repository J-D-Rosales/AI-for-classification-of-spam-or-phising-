# src/data_loader.py

"""
Módulo para cargar y unificar todos los datasets de correos (phishing / no phishing).

La idea es:
- Leer los distintos CSV.
- Normalizar todo a un mismo formato:
    - columna 'text'  -> texto del correo (subject + body, o text_combined)
    - columna 'label' -> 0 (legítimo), 1 (spam/phishing)
- Devolver un único DataFrame listo para usar en el entrenamiento.
"""

from pathlib import Path
import pandas as pd

# Carpeta donde estarán tus CSV
DATA_DIR = Path("data")


def _load_subject_body_csv(filename: str) -> pd.DataFrame:
    """
    Carga CSVs que tienen columnas: 'subject', 'body', 'label'.
    (Enron.csv, Ling.csv)
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    # Aseguramos que las columnas existan
    for col in ["subject", "body", "label"]:
        if col not in df.columns:
            raise ValueError(f"{filename} no tiene la columna requerida: {col}")

    # Reemplazamos valores faltantes por cadenas vacías
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Texto final = subject + body
    df["text"] = df["subject"] + " " + df["body"]

    # Nos quedamos solo con text y label
    df = df[["text", "label"]]

    # Nos aseguramos que label sea entero (0 o 1)
    df["label"] = df["label"].astype(int)

    df["source"] = filename  # opcional: para saber de qué dataset vino
    return df


def _load_full_email_csv(filename: str) -> pd.DataFrame:
    """
    Carga CSVs con más columnas (sender, receiver, date, subject, body, urls, label).
    (SpamAssasin.csv, Nazario.csv, Nigerian_Fraud.csv, CEAS_08.csv)
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    # Aseguramos columnas mínimas
    for col in ["subject", "body", "label"]:
        if col not in df.columns:
            raise ValueError(f"{filename} no tiene la columna requerida: {col}")

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    df["text"] = df["subject"] + " " + df["body"]
    df = df[["text", "label"]]
    df["label"] = df["label"].astype(int)
    df["source"] = filename
    return df


def _load_phishing_email_csv(filename: str) -> pd.DataFrame:
    """
    Carga phishing_email.csv, que ya viene con 'text_combined' + 'label'.
    (Este dataset ya trae subject+body preprocesado.)
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path)

    # Aseguramos columnas
    for col in ["text_combined", "label"]:
        if col not in df.columns:
            raise ValueError(f"{filename} no tiene la columna requerida: {col}")

    df["text"] = df["text_combined"].fillna("")
    df = df[["text", "label"]]
    df["label"] = df["label"].astype(int)
    df["source"] = filename
    return df


def load_all_emails() -> pd.DataFrame:
    """
    Carga y combina TODOS los CSV de correos en un solo DataFrame.

    Devuelve un DataFrame con columnas:
    - text  (string)
    - label (0 = legítimo, 1 = spam/phishing)
    - source (archivo original, opcional)
    """
    dfs = []

    # CSV con subject/body/label
    for fname in ["Enron.csv", "Ling.csv"]:
        dfs.append(_load_subject_body_csv(fname))

    # CSV con columnas extendidas
    for fname in ["SpamAssasin.csv", "Nazario.csv", "Nigerian_Fraud.csv", "CEAS_08.csv"]:
        dfs.append(_load_full_email_csv(fname))

    # CSV especial phishing_email.csv
    dfs.append(_load_phishing_email_csv("phishing_email.csv"))

    # Concatenamos todo
    data = pd.concat(dfs, ignore_index=True)

    # Eliminamos filas con texto vacío (por seguridad)
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]

    # Opcional: eliminar duplicados
    data = data.drop_duplicates(subset=["text", "label"])

    return data


if __name__ == "__main__":
    # Pequeña prueba rápida
    df = load_all_emails()
    print(df.head())
    print(df["label"].value_counts())
