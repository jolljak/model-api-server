from fastapi import APIRouter, UploadFile, File, Form
import asyncio, os, torch
from concurrent.futures import ThreadPoolExecutor

from app.services.storage_service import save_uploaded_file
from app.utils.audio_utils import (
    ensure_tmp_copy,
    transcode_to_wav_mono16k,
    get_media_duration_sec,
    is_silent,
)
from app.services.stt_service import transcribe_file
from app.services.diarization_service import diarize_core
from app.services.summary_service import summarize_text, extract_tasks
from app.utils.merge_utils import assign_speakers, segments_to_text
from app.core.device import get_device

router = APIRouter()


@router.post("/transcribe-diarize")
async def transcribe_diarize(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    createUserId: str = Form(...),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    parallel: bool = Form(True),
):
    data = await file.read()

    file_info = save_uploaded_file(
        data=data,
        original_name=file.filename,
        user_id=createUserId
    )
    abs_path = file_info["abs_path"]

    tmp_path = ensure_tmp_copy(None, data, os.path.splitext(file.filename)[1])

    ok, wav_path, _ = transcode_to_wav_mono16k(tmp_path)
    if not ok:
        return {"resultCode": 0, "message": "오디오 변환 실패"}

    duration = get_media_duration_sec(wav_path)
    if not duration or duration <= 0.5:
        return {"resultCode": 0, "message": "오디오가 너무 짧습니다"}

    if is_silent(wav_path):
        return {"resultCode": 0, "message": "무음 오디오입니다"}

    device = get_device()
    loop = asyncio.get_event_loop()

    if parallel and device.type == "cuda":
        executor = ThreadPoolExecutor(max_workers=2)
        stt_f = loop.run_in_executor(
            executor, lambda: transcribe_file(wav_path, language)
        )
        diar_f = loop.run_in_executor(
            executor, lambda: diarize_core(wav_path, min_speakers, max_speakers)
        )
        stt, diar = await asyncio.gather(stt_f, diar_f)
        torch.cuda.synchronize()
        executor.shutdown(wait=True)
    else:
        stt = transcribe_file(wav_path, language)
        diar = diarize_core(wav_path, min_speakers, max_speakers)

    combined = assign_speakers(stt["segments"], diar["segments"])
    combined_text = segments_to_text(combined)

    # summary = summarize_text(combined_text)
    # tasks = extract_tasks(combined_text)

    return {
        "resultCode": 1,
        "device": str(device),
        "language": stt["language"],
        "duration": stt["duration"],
        "speaker_count": diar["num_speakers"],
        "segments": combined,
        "full_text": stt["text"],
        # "summary": summary,
        # "tasks": tasks,
        # "fileId": _new_file_id, # 녹음 파일 DB ID 반환
    }
