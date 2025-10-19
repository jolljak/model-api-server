# app/main.py
import os
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Literal, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from dotenv import load_dotenv

# ====== 유틸 (공통) ======
from .utils import (
    ensure_tmp_copy,
    transcode_to_wav_mono16k,
    get_media_duration_sec,
    is_silent,
)
from .asr import transcribe_file
from .diarization_utils import build_speaker_map, to_rttm
from pyannote.audio import Pipeline

# ============== 환경설정 ==============
load_dotenv()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def get_device():
    if os.getenv("FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== FastAPI 앱 ==============
app = FastAPI(title="Mina ASR + Diarization (Study Mode)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ============== Pyannote 준비 ==============
HF_TOKEN = os.getenv("HF_TOKEN")
PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=HF_TOKEN)
pipeline.to(get_device())

# ============== 화자 구간 추출 함수 ==============
def diarize_core(audio_path: str, min_speakers=None, max_speakers=None) -> Dict[str, Any]:
    """Pyannote로 화자 구간 분석"""
    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers:
        kwargs["max_speakers"] = int(max_speakers)

    diarization = pipeline({"audio": audio_path}, **kwargs)
    raw_labels = [label for _, _, label in diarization.itertracks(yield_label=True)]
    spk_map = build_speaker_map(raw_labels)

    segments = []
    for turn, _, spk in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": spk_map[spk],
        })

    return {
        "num_speakers": len(set([s["speaker"] for s in segments])),
        "segments": segments,
        "rttm": to_rttm(segments),
    }


# ============== STT + Diar 통합 로직 ==============
def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_speakers(asr_segments: List[dict], diar_segments: List[dict]) -> List[dict]:
    """Whisper 구간과 Pyannote 구간을 시간 겹침 기준으로 매칭"""
    result = []
    for seg in asr_segments:
        a_start, a_end = float(seg.get("start", 0)), float(seg.get("end", 0))
        best_spk, best_overlap = None, 0.0
        for d in diar_segments:
            ov = _overlap(a_start, a_end, d["start"], d["end"])
            if ov > best_overlap:
                best_overlap, best_spk = ov, d["speaker"]

        result.append({
            "start": a_start, "end": a_end,
            "speaker": best_spk or "UNK",
            "text": seg.get("text", "")
        })
    return result


# ============== 메인 엔드포인트 ==============
@app.post("/transcribe-diarize")
async def transcribe_diarize(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    task: Literal["transcribe", "translate"] = Form("transcribe"),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    tmp_path = wav_path = None
    try:
        print("음성파일 분석 시작")
        # 1. 파일 저장
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        tmp_path = ensure_tmp_copy(None, data, os.path.splitext(file.filename)[1])

        # 2. WAV 변환
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_path)
        if not ok:
            raise HTTPException(status_code=415, detail="ffmpeg 변환 실패")

        # 3. 유효성 검사
        if get_media_duration_sec(wav_path) <= 0.5:
            raise HTTPException(status_code=400, detail="오디오 너무 짧음")
        if is_silent(wav_path):
            raise HTTPException(status_code=400, detail="무음 오디오")

        # 4. 병렬 실행 (STT, Diarization)
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)

        stt_future = loop.run_in_executor(executor, lambda: transcribe_file(wav_path, language, task))
        diar_future = loop.run_in_executor(executor, lambda: diarize_core(wav_path, min_speakers, max_speakers))
        stt_result, diar_result = await asyncio.gather(stt_future, diar_future)

        # 5. 매핑
        combined = assign_speakers(stt_result["segments"], diar_result["segments"])

        return {
            "ok": True,
            "device": str(get_device()),
            "language": stt_result.get("language"),
            "duration": stt_result.get("duration"),
            "speaker_count": diar_result["num_speakers"],
            "segments": combined,
            "full_text": stt_result.get("text"),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p): os.remove(p)
            except: pass


# ============== 간단 헬스체크 ==============
@app.get("/healthz")
def health():
    return {"ok": True, "device": str(get_device())}
