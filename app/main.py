from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_loader import load_model, predict
from app.schema import PredictInput

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL later
    allow_methods=["*"],
    allow_headers=["*"],
)
_model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    _model = load_model()
    yield


app = FastAPI(title="Earthquake model API", lifespan=lifespan)


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
