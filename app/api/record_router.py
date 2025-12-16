import os
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form
import torch

# 기존에 만드신 서비스 모듈 활용
from app.services.storage_service import save_uploaded_file
from app.services.stt_service import transcribe_file
from app.services.diarization_service import diarize_core
from app.utils.audio_utils import (
    ensure_tmp_copy,
    transcode_to_wav_mono16k,
    get_media_duration_sec,
    is_silent,
)
from app.utils.merge_utils import assign_speakers, segments_to_text
from app.core.device import get_device
from app.services.processor import MeetingProcessor


router = APIRouter()
processor = MeetingProcessor()

@router.post("/transcribe-diarize")
async def transcribe_diarize(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    createUserId: str = Form(...),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    parallel: bool = Form(True),
):
    """
    업로드된 음성 파일을 저장하고, STT(Whisper)와 화자분리(Pyannote)를 수행합니다.
    """
    tmp_path = None
    wav_path = None

    try:
        # 1. 파일 읽기
        data = await file.read()
        if not data:
            return {"resultCode": 0, "message": "빈 파일입니다."}

        # 2. 파일 저장 (Storage Service 활용)
        # DB 저장이 포함되어 있다면 file_info에 fileId가 있을 것입니다.
        file_info = save_uploaded_file(data, file.filename, createUserId)
        # file_info가 딕셔너리가 아닌 경우(DB 미연결 등) 예외처리 필요할 수 있음
        
        # 3. 오디오 전처리 (wav 16k 변환)
        # 메모리상의 데이터로 임시 파일 생성
        tmp_path = ensure_tmp_copy(None, data, os.path.splitext(file.filename)[1])
        ok, wav_path, _ = transcode_to_wav_mono16k(tmp_path)
        
        if not ok:
            return {"resultCode": 0, "message": "오디오 변환 실패"}

        # 4. 유효성 검사 (길이, 무음)
        duration = get_media_duration_sec(wav_path)
        if not duration or duration <= 0.5:
            return {"resultCode": 0, "message": "오디오가 너무 짧습니다."}
        
        if is_silent(wav_path):
            return {"resultCode": 0, "message": "무음 파일입니다."}

        # 5. 분석 수행 (병렬 처리)
        device = get_device()
        loop = asyncio.get_event_loop()

        # GPU 사용 가능 시 병렬 실행
        if parallel and device.type == "cuda":
            executor = ThreadPoolExecutor(max_workers=2)
            # STT 실행
            stt_future = loop.run_in_executor(
                executor, lambda: transcribe_file(wav_path, language)
            )
            # Diarization 실행
            diar_future = loop.run_in_executor(
                executor, lambda: diarize_core(wav_path, min_speakers, max_speakers)
            )
            
            stt_res, diar_res = await asyncio.gather(stt_future, diar_future)
            executor.shutdown(wait=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # 순차 실행
            stt_res = transcribe_file(wav_path, language)
            diar_res = diarize_core(wav_path, min_speakers, max_speakers)

        # 6. 결과 병합 (STT + Speaker)
        combined = assign_speakers(stt_res["segments"], diar_res["segments"])
        formatted_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in combined])
        full_text = stt_res.get("text", "")
        analysis = await processor.analyze_transcript(formatted_text)
        summary_text = analysis.summary
        tasks_payload = [
            {
                "speaker": st.speaker,
                "items": [
                    {
                        "업무설명": it.업무설명,
                        "priority": it.priority
                    }
                    for it in st.items
                ]
            }
            for st in analysis.tasks
        ]


        print("모델 요약 결과 확인", summary_text)
        print("모델 할 일 결과 확인", tasks_payload)

        return {
            "resultCode": 1,
            "fileId": file_info.get("fileId", 0) if isinstance(file_info, dict) else 0,
            "duration": stt_res.get("duration"),
            "speaker_count": diar_res.get("num_speakers"),
            "segments": combined,
            "full_text": full_text,
            "formatted_text": formatted_text,
            "summary": summary_text,
            "tasks": tasks_payload
        }

    except Exception as e:
        traceback.print_exc()
        return {"resultCode": 0, "message": f"Analysis Error: {str(e)}"}

    finally:
        # 임시 파일 정리
        for p in (tmp_path, wav_path):
            try:
                if p and os.path.exists(p): os.remove(p)
            except Exception: pass