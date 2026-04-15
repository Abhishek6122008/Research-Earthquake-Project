import io
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.model_loader import load_model, predict

logger = logging.getLogger(__name__)
model = None

router = APIRouter()


def _get_model():
    global model
    if model is None:
        logger.info("Loading model...")
        try:
            model = load_model()
            logger.info("Model loaded successfully")
        except FileNotFoundError as exc:
            logger.error("Model file not found: %s", exc)
            raise HTTPException(status_code=503, detail="Model file not found") from exc
    return model


@router.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        active_model = _get_model()
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            raise HTTPException(
                status_code=400, detail="CSV must contain at least one numeric column"
            )
        flat = numeric.to_numpy(dtype=float).flatten().tolist()
        prediction = predict(active_model, flat)
        return {"prediction": prediction}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
