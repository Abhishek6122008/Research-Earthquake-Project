from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_model, predict
from app.schema import PredictInput

_model = None
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    logger.info("Loading model...")
    try:
        _model = load_model()
        logger.info("Model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        _model = None
    except Exception as exc:  # pragma: no cover - startup protection
        logger.exception("Model failed to load: %s", exc)
        _model = None
    yield


app = FastAPI(title="Earthquake model API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_route(body: PredictInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"prediction": predict(_model, body.input)}

from app.file_predict import router as file_router
app.include_router(file_router)
