from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from pathlib import Path
import os

from src.predict import SpamDetector

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.7"))
detector = SpamDetector(threshold=THRESHOLD)

app = FastAPI(title="Spam Detector")

# Sirve / y /static
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/health")
async def health():
    return {"ok": True, "threshold": THRESHOLD}

# ====== handler global para cualquier excepción ======
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        {"error": "internal_error", "detail": str(exc)},
        status_code=500
    )

@app.post("/predict")
async def predict(payload: dict):
    try:
        text = payload.get("text", "")
        if not text or not isinstance(text, str):
            return JSONResponse({"error": "Falta 'text'."}, status_code=400)
        return detector.predict(text)
    except Exception as e:
        # captura errores del modelo y responde JSON
        return JSONResponse({"error": "prediction_failed", "detail": str(e)}, status_code=500)
