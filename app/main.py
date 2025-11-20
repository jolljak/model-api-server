from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.stt_router import router as stt_router
from app.api.health_router import router as health_router

app = FastAPI(title="Mina ASR + Diarization (Modular Version)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(stt_router)
app.include_router(health_router)
