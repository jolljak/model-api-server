from app.models.diarization_model import get_diarization_pipeline
from app.utils.diarization_utils import build_speaker_map, to_rttm

def diarize_core(path: str, min_speakers=None, max_speakers=None):
    pipeline = get_diarization_pipeline()

    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers:
        kwargs["max_speakers"] = int(max_speakers)

    diar = pipeline({"audio": path}, **kwargs)

    raw = [lbl for _, _, lbl in diar.itertracks(yield_label=True)]
    spk_map = build_speaker_map(raw)

    segments = []
    for turn, _, spk in diar.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": spk_map[spk]
        })  

    return {
        "num_speakers": len(set(s["speaker"] for s in segments)),
        "segments": segments,
        "rttm": to_rttm(segments)
    }
