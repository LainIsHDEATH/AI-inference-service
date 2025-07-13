from fastapi import APIRouter, Request
from app.schemas.rl import RlModelDtoRequest, RlModelDtoResponse
from app.services.rl_service import rl_compute

router = APIRouter()

@router.post("/compute", response_model=RlModelDtoResponse)
def compute(request: Request, req: RlModelDtoRequest):
    return rl_compute(request.app, req)
