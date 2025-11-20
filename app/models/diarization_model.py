from pyannote.audio import Pipeline
from app.core.config import HF_TOKEN
from app.core.device import get_device

_pipeline = None

def get_diarization_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        _pipeline.to(get_device())
    return _pipeline
