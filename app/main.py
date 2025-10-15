# app/main.py
import os
import io
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict, Literal, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
import torch

# Whisper utils (이미 프로젝트에 있는 유틸 사용)
from .utils import (
    ensure_tmp_copy,
    get_media_duration_sec,
    is_silent,
    transcode_to_wav_mono16k,
)
from .asr import transcribe_file

# Pyannote
from pyannote.audio import Pipeline
from .diarization_utils import build_speaker_map, to_rttm

# ============== 공통 환경설정 ==============
load_dotenv()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def get_device() -> torch.device:
    # 테스트로 CPU 강제하고 싶으면 환경변수 FORCE_CPU=1로 설정
    if os.getenv("FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디버그(원하면 주석):
# try:
#     print("[torch] CUDA (built):", getattr(torch.version, "cuda", None))
#     print("[torch] cuDNN:", torch.backends.cudnn.version())
#     print("[torch] CUDA available:", torch.cuda.is_available())
# except Exception as _e:
#     print("[torch] env print error:", _e)

# ============== FastAPI 앱 ==============
app = FastAPI(title="Mina ASR + Diarization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ============== Whisper (STT) ==============
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
    tmp_in_path = None
    wav_path = None
    try:
        # 업로드 읽기 → 임시 저장
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        file_bytes = await file.read()
        tmp_in_path = ensure_tmp_copy(None, file_bytes, suffix)

        # 변환
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_in_path)
        if not ok:
            raise HTTPException(status_code=415, detail=f"ffmpeg 변환 실패: {log[:4000]}")

        # 길이/무음 검사
        raw_dur = get_media_duration_sec(wav_path)
        if raw_dur is not None and raw_dur <= 0.5:
            raise HTTPException(status_code=400, detail="오디오 길이가 너무 짧습니다(0.5초 이하).")
        if is_silent(wav_path):
            raise HTTPException(status_code=400, detail="업로드된 오디오가 무음으로 감지되었습니다.")

        # STT
        result = transcribe_file(wav_path, language=language, task=task)

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

    except HTTPException:
        raise
    except Exception as e:
        # 어디서 터졌는지 콘솔에 traceback 남기기
        print("[/transcribe] ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"transcribe 실패: {e}")
    finally:
        for p in (tmp_in_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# ============== Pyannote (Diarization) ==============
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
    max_speakers: Optional[int] = None,
) -> Dict[str, Any]:
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    # 핵심: 파이프라인에는 dict로 전달 (torchaudio가 확실히 경로로 인식)
    diarization = _pipeline({"audio": audio_path}, **kwargs)

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

# ==== diarize_endpoint에서 WAV로 변환 후 그 경로를 넘기도록 수정 ====
@app.post("/diarize")
async def diarize_endpoint(
    file: UploadFile = File(..., description="오디오/영상 파일"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    tmp_path = None
    wav_path = None
    try:
        # 원본 저장
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        ext = os.path.splitext(file.filename or "")[1] or ".bin"
        tmp_path = ensure_tmp_copy(None, content, ext)

        # 반드시 WAV(mono,16k)로 변환
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_path)
        if not ok or not wav_path or not os.path.exists(wav_path):
            raise HTTPException(status_code=415, detail=f"ffmpeg 변환 실패: {str(log)[:4000]}")

        # (선택) 길이/무음 방어
        dur = get_media_duration_sec(wav_path)
        if dur is not None and dur <= 0.5:
            raise HTTPException(status_code=400, detail="오디오 길이가 너무 짧습니다(0.5초 이하).")
        if is_silent(wav_path):
            raise HTTPException(status_code=400, detail="업로드된 오디오가 무음으로 감지되었습니다.")

        # 변환한 wav_path를 파이프라인에 전달
        result = diarize_core(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)

        return {
            "ok": True,
            "device": str(get_device()),
            "pipeline": PIPELINE_NAME,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("[/diarize] ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"diarization 실패: {e}")
    finally:
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

# ============== STT + Diar 통합 ==============
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_speakers_by_overlap(
    asr_segments: List[dict],
    diar_segments: List[dict],
) -> List[dict]:
    """
    ASR 세그먼트(start,end,text)와 DIAR 세그먼트(start,end,speaker)를
    시간 겹침 최대치 기준으로 매칭하여 speaker를 부여한다.
    """
    out = []
    for seg in asr_segments:
        a_start = float(seg.get("start", 0.0))
        a_end = float(seg.get("end", a_start))
        text = seg.get("text", "")

        best_speaker = None
        best_ov = 0.0
        for d in diar_segments:
            ov = _overlap(a_start, a_end, float(d["start"]), float(d["end"]))
            if ov > best_ov:
                best_ov = ov
                best_speaker = d["speaker"]

        out.append({
            "start": round(a_start, 3),
            "end": round(a_end, 3),
            "speaker": best_speaker or "UNK",
            "text": text,
        })
    return out

@app.post("/transcribe-diarize")
async def transcribe_diarize_endpoint(
    file: UploadFile = File(..., description="오디오/영상 파일"),
    language: str = Form("auto"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    parallel: bool = Form(True),  # GPU 여유 없으면 False로 보내서 직렬 실행
):
    tmp_path = None
    wav_path = None
    try:
        # 1) 업로드 저장
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        ext = os.path.splitext(file.filename or "")[1] or ".bin"
        tmp_path = ensure_tmp_copy(None, content, ext)

        # 2) WAV(mono,16k) 한 번만 변환
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_path)
        if not ok or not wav_path or not os.path.exists(wav_path):
            raise HTTPException(status_code=415, detail=f"ffmpeg 변환 실패: {str(log)[:4000]}")

        # 3) 방어로직
        dur = get_media_duration_sec(wav_path)
        if dur is not None and dur <= 0.5:
            raise HTTPException(status_code=400, detail="오디오 길이가 너무 짧습니다(0.5초 이하).")
        if is_silent(wav_path):
            raise HTTPException(status_code=400, detail="업로드된 오디오가 무음으로 감지되었습니다.")

        # 4) STT & DIAR 실행
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)

        def _do_stt():
            return transcribe_file(wav_path, language=language, task=task)

        def _do_diar():
            return diarize_core(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)

        if parallel:
            stt_future = loop.run_in_executor(executor, _do_stt)
            diar_future = loop.run_in_executor(executor, _do_diar)
            stt_result, diar_result = await asyncio.gather(stt_future, diar_future)
        else:
            stt_result = _do_stt()
            diar_result = _do_diar()

        # 5) 화자 매핑(세그먼트 겹침)
        asr_segments = stt_result.get("segments", [])
        diar_segments = diar_result.get("segments", [])
        diarized_segments = assign_speakers_by_overlap(asr_segments, diar_segments)

        return {
            "ok": True,
            "device": str(get_device()),
            "pipeline": {
                "diarization": PIPELINE_NAME,
                "stt_model": stt_result.get("model_name"),
            },
            "duration_sec": stt_result.get("duration"),
            "language": stt_result.get("language"),
            "language_probability": stt_result.get("language_probability"),
            "speakers": {
                "count": diar_result.get("num_speakers"),
            },
            "rttm": diar_result.get("rttm"),
            "segments": diarized_segments,  # [{start,end,speaker,text}]
            "text": stt_result.get("text", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("[/transcribe-diarize] ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"transcribe+diarize 실패: {e}")
    finally:
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass

# ============== 헬스체크 ==============
@app.get("/healthz")
def healthz():
    return {"ok": True, "msg": "ready"}

@app.get("/health/pyannote")
def health_pyannote():
    return {"ok": True, "device": str(get_device()), "pipeline": PIPELINE_NAME}

@app.get("/ping")
def ping():
    return {"pong": True, "device": str(get_device()), "pipeline": PIPELINE_NAME}

@app.get("/test")
def test():
    return 'cicd test success'