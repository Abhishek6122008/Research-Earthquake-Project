import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.model_loader import load_model, predict

model = load_model()

router = APIRouter()


@router.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            raise HTTPException(
                status_code=400, detail="CSV must contain at least one numeric column"
            )
        flat = numeric.to_numpy(dtype=float).flatten().tolist()
        prediction = predict(model, flat)
        return {"prediction": prediction}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
