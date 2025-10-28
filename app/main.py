import os
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import torch
from app.database import get_connection
import time, uuid

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

# ============== GPU 디바이스 자동 감지 함수 ==============
def get_device() -> torch.device:
    """CUDA가 가능하면 GPU로 강제"""
    if os.getenv("FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== FastAPI 앱 ==============
app = FastAPI(title="Mina ASR + Diarization (Study Mode)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Pyannote 준비 ==============
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("환경변수 HF_TOKEN이 설정되어야 합니다 (Hugging Face 토큰).")
PIPELINE_NAME = "pyannote/speaker-diarization-3.1"
print(f"[INFO] Loading Pyannote pipeline: {PIPELINE_NAME}")
pipeline = Pipeline.from_pretrained(PIPELINE_NAME, use_auth_token=HF_TOKEN)
pipeline.to(get_device())
print("[INFO] Pyannote loaded successfully.")

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
            "start": a_start,
            "end": a_end,
            "speaker": best_spk or "UNK",
            "text": seg.get("text", "")
        })
    return result


# ============== 메인 엔드포인트 ==============
@app.post("/transcribe-diarize")
async def transcribe_diarize(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    createUserId: str = Form(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    parallel: bool = Form(True),  # 병렬 실행 여부 (GPU 여유 없으면 False로 보냄)
):
    tmp_path = wav_path = None
    try:
        db = get_connection()
        # 파일 저장
        data = await file.read()
        if not data:
            return {"resultCode": 0, "message": "빈 파일입니다. 업로드를 다시 확인하세요."}
         # 1) 저장 경로 준비: app/uploads/{createUserId}/{YYYYMMDD}/
        uploads_root = os.path.join(os.path.dirname(__file__), "uploads")
        datedir = time.strftime("%Y%m%d")
        user_dir = os.path.join(uploads_root, createUserId, datedir)
        os.makedirs(user_dir, exist_ok=True)

        # 2) 원본 이름/확장자, 고유 저장 파일명 생성
        orig_name = os.path.basename(file.filename)
        _, ext_with_dot = os.path.splitext(orig_name)
        ext = ext_with_dot.lstrip(".").lower()
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex}{ext_with_dot}"
        abs_path = os.path.join(user_dir, unique_name)

        # 3) 디스크에 저장 (방금 읽은 data 사용)
        with open(abs_path, "wb") as f:
            f.write(data)
        file_size = os.path.getsize(abs_path)

        # 4) DB INSERT (상대경로 저장)
        rel_path = os.path.relpath(abs_path, uploads_root).replace("\\", "/")
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO dbo.TB_MINA_FILE_L
              (filePath, fileSize, fileName, fileExt, createUserId)
            OUTPUT INSERTED.fileId
            VALUES (?, ?, ?, ?, ?)
            """,
            (rel_path, file_size, orig_name, ext, createUserId),
        )
        _new_file_id = cur.fetchone()[0]  # 필요하면 사용
        db.commit()

        tmp_path = ensure_tmp_copy(None, data, os.path.splitext(file.filename)[1])

        # WAV 변환
        ok, wav_path, log = transcode_to_wav_mono16k(tmp_path)
        if not ok:
            return {"resultCode": 0, "message": "오디오 변환 실패 (ffmpeg 에러)"}

        # 유효성 검사
        duration = get_media_duration_sec(wav_path)
        if duration <= 0.5:
            return {"resultCode": 0, "message": "오디오가 너무 짧습니다. (0.5초 미만)"}

        if is_silent(wav_path):
            return {"resultCode": 0, "message": "무음 오디오입니다. 다시 녹음해주세요."}

        # GPU 기반 실행
        device = get_device()
        loop = asyncio.get_event_loop()

        if parallel and device.type == "cuda":
            executor = ThreadPoolExecutor(max_workers=2)
            stt_future = loop.run_in_executor(executor, lambda: transcribe_file(wav_path, language))
            diar_future = loop.run_in_executor(executor, lambda: diarize_core(wav_path, min_speakers, max_speakers))
            stt_result, diar_result = await asyncio.gather(stt_future, diar_future)
        else:
            print("[INFO] Running sequentially to reduce GPU contention.")
            stt_result = transcribe_file(wav_path, language)
            diar_result = diarize_core(wav_path, min_speakers, max_speakers)

        # 매핑
        combined = assign_speakers(stt_result["segments"], diar_result["segments"])

        print("분석 완료 (GPU 모드)")
        return {
            "resultCode": 1,
            "device": str(device),
            "language": stt_result.get("language"),
            "duration": stt_result.get("duration"),
            "speaker_count": diar_result["num_speakers"],
            "segments": combined,
            "full_text": stt_result.get("text"),
        }

    except Exception as e:
        traceback.print_exc()
        return {"resultCode": 0, "message": f"분석 중 오류 발생: {str(e)}"}

    finally:
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ============== 간단 헬스체크 ==============
@app.get("/healthz")
def health():
    return {"resultCode": 1, "device": str(get_device()), "cuda_available": torch.cuda.is_available()}
