# main.py

from src.predict import SpamDetector


def main():
    # Cambia el threshold si quieres ser más o menos estricto (0.7, 0.8, etc.)
    detector = SpamDetector(threshold=0.7)

    print("=== Detector de Spam/Phishing (INGLÉS) ===")
    print("Escribe el texto del correo en inglés que quieres analizar.")
    print("Escribe 'salir' para terminar.\n")
    print(f"(Umbral actual para SPAM: {detector.threshold * 100:.1f}%)\n")

    while True:
        texto = input("Correo> ")

        if texto.strip().lower() == "salir":
            print("Saliendo del detector. ¡Hasta luego!")
            break

        resultado = detector.predict(texto)

        es_spam = resultado["is_spam"]
        prob = resultado["spam_probability"] * 100
        thr = resultado["threshold"] * 100

        etiqueta = "SPAM/PHISHING" if es_spam else "LEGITIMATE"

        print(f"Result: {etiqueta}")
        print(f"Spam probability: {prob:.2f}% (threshold: {thr:.2f}%)\n")


if __name__ == "__main__":
    main()
