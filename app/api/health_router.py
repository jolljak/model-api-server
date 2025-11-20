from fastapi import APIRouter
import torch
from app.core.device import get_device

router = APIRouter()

@router.get("/healthz")
def health():
    return {
        "resultCode": 1,
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available()
    }
