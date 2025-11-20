from faster_whisper import WhisperModel
from app.core.config import ASR_MODEL, ASR_COMPUTE

_model = None

def get_asr_model():
    global _model
    if _model is None:
        _model = WhisperModel(ASR_MODEL, compute_type=ASR_COMPUTE)
    return _model
