import json, logging, math
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple
import torch as _T
import dto

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path("models")
SEQ_LENGTH = 30
FEATURES   = 4

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=FEATURES, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):           # (N,L,F)
        out, _ = self.lstm(x)
        return self.fc(out[:,-1]).squeeze(-1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lstm_models: Dict[int, dict] = {}
    for ckpt in BASE_DIR.rglob("checkpoint.joblib"):
        meta = joblib.load(ckpt)
        mid = int(ckpt.parent.name.split("-")[1])

        net = TemperatureLSTM(
            input_size = FEATURES,
            hidden_size= meta["hidden_size"],
            num_layers = meta["num_layers"],
        )
        net.load_state_dict(meta["state_dict"])
        net.eval()

        lstm_models[mid] = {
            "net":      net,
            "x_mean":   torch.tensor(meta["x_mean"]),
            "x_std":    torch.tensor(meta["x_std"]),
            "y_mean":    meta["y_mean"],
            "y_std":     meta["y_std"],
            "seq_len":   meta["seq_length"],
        }

    app.state.lstm_models = lstm_models

    rl_tables: Dict[int, np.ndarray] = {}
    for table_path in BASE_DIR.rglob("q_table.npy"):
        mid = int(table_path.parent.name.split("-")[1])
        rl_tables[mid] = np.load(table_path)
    app.state.rl_tables = rl_tables
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/lstm/predict", response_model=LstmModelDtoResponse)
def lstm_predict(req: LstmModelDtoRequest):
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

    feats = _T.tensor([[
        [d.tempIn, d.tempOut, d.tempSetpoint, d.heaterPower]
        for d in data
    ]], dtype=_T.float32)

    feats_n = (feats - x_mean) / x_std

    with torch.no_grad():
        pred_n = net(feats_n).item()

    pred = pred_n * y_std + y_mean
    return {"predictedTemp": round(pred, 2)}

@app.post("/rl/compute", response_model=RlModelDtoResponse)
def rl_compute(req: RlModelDtoRequest):
    q = app.state.rl_tables.get(req.modelId)
    if q is None:
        raise HTTPException(404, "Model not found")
    q = app.state.rl_tables.get(req.modelId)
    if q is None:
        raise HTTPException(404, "Model not found")
    def bin(v, lo, hi, n=20): return int(np.clip((v-lo)/(hi-lo)*n, 0, n-1))
    s_r, s_o = bin(req.tempIn,10,30), bin(req.tempOut,-20,40)
    a_bin = q[s_r, s_o].argmax()
    pct   = a_bin/(q.shape[-1]-1)
    logging.info("Predicted temperature is %f", round(pct*100, 2))
    print("Calculated temmperature is %f", round(pct*100, 2))
    return {"heaterPower": round(pct*100, 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7003, reload=True)