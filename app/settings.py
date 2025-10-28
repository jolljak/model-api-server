from pathlib import Path

# 예: 프로젝트 루트/app/uploads 에 저장
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)