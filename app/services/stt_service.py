from app.models.asr_model import get_asr_model
from app.core.config import ASR_BEAM

def transcribe_file(wav_path: str, language: str):
    model = get_asr_model()

    seg_gen, info = model.transcribe(
        wav_path,
        beam_size=ASR_BEAM,
        language=None if language == "auto" else language,
        vad_filter=True,
    )

    segments = [
        {
            "id": i,
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
        }
        for i, s in enumerate(seg_gen)
    ]

    return {
        "language": info.language,
        "duration": info.duration,
        "segments": segments,
        "text": "".join(s["text"] for s in segments).strip(),
    }
