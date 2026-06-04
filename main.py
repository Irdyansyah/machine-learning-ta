import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from model.predictor import VARKPredictor

VALID_STYLES = {'visual', 'auditory', 'read_write', 'kinesthetic'}
ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'

predictor: VARKPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = VARKPredictor()
    yield


app = FastAPI(title='VARK ML Service', lifespan=lifespan)


class PredictRequest(BaseModel):
    answers: list[str]

    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v: list[str]) -> list[str]:
        if len(v) != 8:
            raise ValueError(f'Expected 8 answers, got {len(v)}')
        for ans in v:
            if ans not in VALID_STYLES:
                raise ValueError(f'Invalid answer "{ans}". Must be one of {VALID_STYLES}')
        return v


class PredictResponse(BaseModel):
    dominant_style: str
    probabilities: dict[str, float]
    scores: dict[str, int]


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'model_loaded': predictor is not None}


@app.get('/model/info')
def model_info() -> dict:
    metrics_path = ARTIFACTS_DIR / 'metrics.json'
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail='Metrics not found. Run train.py first.')
    with open(metrics_path) as f:
        return json.load(f)


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    result = predictor.predict(req.answers)
    return PredictResponse(**result)
