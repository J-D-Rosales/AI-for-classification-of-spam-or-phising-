# src/preprocess.py

"""
Módulo de preprocesamiento de texto para correos en INGLÉS.

Usamos TF-IDF de palabras (word-level):

- lowercase: convierte todo a minúsculas.
- stop_words="english": quita palabras muy comunes en inglés (the, and, of...).
- ngram_range=(1, 2): usa unigramas y bigramas (palabras individuales y pares).
- min_df y max_df: ayudan a limpiar ruido.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """
    Crea y devuelve un TfidfVectorizer para texto en inglés.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",   # asumimos correos en inglés
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=5,               # ignorar palabras muy raras (aparecen <5 veces)
        max_df=0.95,            # ignorar palabras demasiado frecuentes (>95% docs)
        sublinear_tf=True       # usa tf sublineal (log(1 + tf)), a veces mejora
    )
    return vectorizer
