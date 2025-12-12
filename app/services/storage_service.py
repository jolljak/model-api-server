import os
import time
import uuid
from app.core.database import get_connection

UPLOAD_ROOT = os.path.join(os.path.dirname(__file__), "..", "uploads")

def save_uploaded_file(data: bytes, original_name: str, user_id: str):
    """
    - uploads/{userId}/{YYYYMMDD}/{unique_filename.ext} 저장
    - TB_MINA_FILE_L INSERT
    - fileId 반환
    """
    datedir = time.strftime("%Y%m%d")
    user_dir = os.path.join(UPLOAD_ROOT, user_id, datedir)
    os.makedirs(user_dir, exist_ok=True)

    # 파일명/확장자
    _, ext = os.path.splitext(original_name)
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex}{ext}"
    abs_path = os.path.join(user_dir, unique_name)

    # 저장
    with open(abs_path, "wb") as f:
        f.write(data)

    file_size = os.path.getsize(abs_path)

    # DB 저장
    rel_path = os.path.relpath(abs_path, UPLOAD_ROOT).replace("\\", "/")
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO dbo.TB_MINA_FILE_L
          (filePath, fileSize, fileName, fileExt, createUserId)
        OUTPUT INSERTED.fileId
        VALUES (?, ?, ?, ?, ?)
        """,
        (rel_path, file_size, original_name, ext.lstrip("."), user_id),
    )
    file_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return {
        "file_id": file_id,
        "abs_path": abs_path,
        "rel_path": rel_path,
        "file_size": file_size,
    }
