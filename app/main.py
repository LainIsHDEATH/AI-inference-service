from fastapi import FastAPI
from app.routers.lstm_router import router as lstm_router
from app.routers.rl_router import router as rl_router
from app.services.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(lstm_router, prefix="/lstm", tags=["lstm"])
app.include_router(rl_router, prefix="/rl", tags=["rl"])

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7003, reload=True)