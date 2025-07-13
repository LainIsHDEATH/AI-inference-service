from fastapi import APIRouter, Request
from app.schemas.lstm import LstmModelDtoRequest, LstmModelDtoResponse
from app.services.lstm_service import lstm_predict

router = APIRouter()

@router.post("/predict", response_model=LstmModelDtoResponse)
def predict(request: Request, req: LstmModelDtoRequest):
    return lstm_predict(request.app, req)
