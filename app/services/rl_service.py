import numpy as np
import logging
from fastapi import HTTPException
from app.schemas.rl import RlModelDtoRequest, RlModelDtoResponse

def rl_compute(app, req: RlModelDtoRequest):
    q = app.state.rl_tables.get(req.modelId)
    if q is None:
        raise HTTPException(404, "Model not found")
    def bin(v, lo, hi, n=101): return int(np.clip((v-lo)/(hi-lo)*n, 0, n-1))
    s_r, s_o = bin(req.tempIn,10,30), bin(req.tempOut,-10,30)
    a_bin = q[s_r, s_o].argmax()
    pct   = a_bin/(q.shape[-1]-1)
    print("Heater power: %f", round(pct*100, 2))
    logging.info("Heater power: %f", round(pct*100, 2))
    return RlModelDtoResponse(heaterPower=round(pct*100, 2))
