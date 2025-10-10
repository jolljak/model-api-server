# app/main.py
import os
from typing import Optional, Any, Dict, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import torch

# Whisper 유틸
from .utils import (
    ensure_tmp_copy,
    get_media_duration_sec,
    is_silent,
    transcode_to_wav_mono16k,
    save_upload_to_tmp,  # diarize에서 사용
)
from .asr import transcribe_file

# Pyannote 유틸
from pyannote.audio import Pipeline
from .diarization_utils import build_speaker_map, to_rttm

# =========================
# 공통 환경설정
# =========================
load_dotenv()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 디버그: CUDA/ cuDNN 정보 출력 (선택)
# try:
#     print("CUDA version (built):", getattr(torch.version, "cuda", None))
#     print("cuDNN version:", torch.backends.cudnn.version())
#     print("CUDA available:", torch.cuda.is_available())
# except Exception as _e:
#     print("Torch env print error:", _e)

# =========================
# FastAPI 앱
# =========================
app = FastAPI(title="Mina ASR + Diarization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요시 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Whisper (STT) 섹션
# =========================
class TranscribeResponse(BaseModel):
    ok: bool
    text: str
    segments: list[dict]
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration_sec: Optional[float] = None
    model: Optional[str] = None
    task: Literal["transcribe", "translate"] = "transcribe"
    detail: Optional[str] = None

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="오디오/영상 파일(mp3, wav, webm, ogg 등)"),
    language: str = Form("auto"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
):
    """
    멀티파트 입력을 받아 Whisper로 STT 수행.
    - language: "auto" | "ko" | "en" …
    - task: "transcribe"(원문언어 그대로) | "translate"(영어로 번역)
    """
    tmp_in_path = None
    wav_path = None

    try:
        # 1) 업로드 파일 → 임시 저장
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        file_bytes = await file.read()
        tmp_in_path = ensure_tmp_copy(None, file_bytes, suffix)

        # 2) Whisper 친화 포맷으로 변환 (mono16k wav)
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_in_path)
        if not ok:
            raise HTTPException(status_code=415, detail=f"ffmpeg 변환 실패: {log[:4000]}")

        # 3) 길이 검사
        raw_dur = get_media_duration_sec(wav_path)  # 변환된 wav 기준
        if raw_dur is not None and raw_dur <= 0.5:
            raise HTTPException(
                status_code=400,
                detail="오디오 길이가 너무 짧습니다(0.5초 이하). 다시 녹음해주세요."
            )

        # 3-1) 무음 검사
        if is_silent(wav_path):
            raise HTTPException(
                status_code=400,
                detail="업로드된 오디오가 무음으로 감지되었습니다. 다시 녹음해주세요."
            )

        # 4) STT
        result = transcribe_file(wav_path, language=language, task=task)

        # 5) 응답
        return TranscribeResponse(
            ok=True,
            text=result["text"],
            segments=result["segments"],
            language=result.get("language"),
            language_probability=result.get("language_probability"),
            duration_sec=result.get("duration"),
            model=result.get("model_name"),
            task=task,
        )

    finally:
        # 임시파일 정리
        for p in (tmp_in_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# =========================
# Pyannote (Diarization) 섹션
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN 환경변수가 필요합니다. (Hugging Face token)")

PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
# print(f"[pyannote] Loading pipeline: {PIPELINE_NAME}")
_pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=HF_TOKEN)
_pipeline.to(get_device())
# print(f"[pyannote] Loaded to device: {get_device()}")

def diarize_core(
    audio_path: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict[str, Any]:
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    diarization = _pipeline(audio_path, **kwargs)

    raw_labels = [label for _, _, label in diarization.itertracks(yield_label=True)]
    spk_map = build_speaker_map(raw_labels)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": spk_map[speaker],
        })

    return {
        "num_speakers": len(set([s["speaker"] for s in segments])),
        "segments": segments,
        "rttm": to_rttm(segments),
    }

@app.post("/diarize")
async def diarize_endpoint(
    file: UploadFile = File(..., description="오디오/영상 파일"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    tmp_path = None
    try:
        # 업로드 파일 임시 저장
        tmp_path = save_upload_to_tmp(file)
        result = diarize_core(tmp_path, min_speakers=min_speakers, max_speakers=max_speakers)
        return {
            "ok": True,
            "device": str(get_device()),   # JSON 직렬화 가능하게 str로
            "pipeline": PIPELINE_NAME,
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diarization 실패: {e}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# =========================
# 헬스체크
# =========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "msg": "ready"}

@app.get("/health/pyannote")
def health_pyannote():
    return {"ok": True, "device": str(get_device()), "pipeline": PIPELINE_NAME}

@app.get("/ping")
def ping():
    return {"pong": True, "device": str(get_device()), "pipeline": PIPELINE_NAME}
