import torch
from fastapi import HTTPException
from app.schemas.lstm import LstmModelDtoRequest, LstmModelDtoResponse

def lstm_predict(app, req: LstmModelDtoRequest):
    bundle = app.state.lstm_models.get(req.modelId)
    if not bundle:
        raise HTTPException(404, "Model not found")
    net     = bundle["net"]
    x_mean  = bundle["x_mean"]
    x_std   = bundle["x_std"]
    y_mean  = bundle["y_mean"]
    y_std   = bundle["y_std"]
    L       = bundle["seq_len"]

    window = req.sensorDataDTOList
    if len(window) < L:
        raise HTTPException(422, f"Need ≥{L} sensor records")

    data = window[-L:]
    feats = torch.tensor([[
        [d.tempIn, d.tempOut, d.tempSetpoint, d.heaterPower]
        for d in data
    ]], dtype=torch.float32)
    feats_n = (feats - x_mean) / x_std

    with torch.no_grad():
        pred_n = net(feats_n).item()

    pred = pred_n * y_std + y_mean
    return LstmModelDtoResponse(predictedTemp=round(pred, 2))
