import joblib
import torch
import asyncio
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from app.models.lstm_model import TemperatureLSTM
from app.config import BASE_DIR

FEATURES = 4

def load_models():
    lstm_models = {}
    for ckpt in BASE_DIR.rglob("checkpoint.joblib"):
        meta = joblib.load(ckpt)
        mid = int(ckpt.parent.name.split("-")[1])
        net = TemperatureLSTM(
            input_size=FEATURES,
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
        )
        net.load_state_dict(meta["state_dict"])
        net.eval()
        lstm_models[mid] = {
            "net": net,
            "x_mean": torch.tensor(meta["x_mean"]),
            "x_std": torch.tensor(meta["x_std"]),
            "y_mean": meta["y_mean"],
            "y_std": meta["y_std"],
            "seq_len": meta["seq_length"],
        }

    rl_tables = {}
    for table_path in BASE_DIR.rglob("q_table.npy"):
        mid = int(table_path.parent.name.split("-")[1])
        rl_tables[mid] = np.load(table_path)
    return lstm_models, rl_tables

async def periodic_model_refresh(app, interval_sec=60):
    while True:
        try:
            lstm_models, rl_tables = load_models()
            app.state.lstm_models = lstm_models
            app.state.rl_tables = rl_tables
            print("[INFO] Models refreshed")
        except Exception as e:
            print(f"[ERROR] Error while refreshing models: {e}")
        await asyncio.sleep(interval_sec)

@asynccontextmanager
async def lifespan(app):
    lstm_models, rl_tables = load_models()
    app.state.lstm_models = lstm_models
    app.state.rl_tables = rl_tables

    refresh_task = asyncio.create_task(periodic_model_refresh(app, interval_sec=60))

    yield

#     refresh_task.cancel()
#     try:
#         await refresh_task
#     except asyncio.CancelledError:
#         pass
