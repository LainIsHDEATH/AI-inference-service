from fastapi import FastAPI
from app.routers.lstm_router import router as lstm_router
from app.routers.rl_router import router as rl_router
from app.services.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(lstm_router, prefix="/lstm", tags=["lstm"])
app.include_router(rl_router, prefix="/rl", tags=["rl"])