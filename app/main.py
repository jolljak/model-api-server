import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api import mcp_router, record_router
from app.api.stt_router import router as legacy_stt_router
from app.api.health_router import router as health_router

load_dotenv()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

app = FastAPI(
    title="Mina AI: Hybrid Server (Legacy + MCP)",
    description="Support both Transformers(Legacy) and Ollama/LangChain(MCP)",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.include_router(health_router)

app.include_router(mcp_router.router)
app.include_router(record_router.router)

app.include_router(legacy_stt_router, prefix="/api/legacy", tags=["Legacy"])

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)