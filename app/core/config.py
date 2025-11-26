import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

ASR_MODEL = os.getenv("ASR_MODEL", "base")
ASR_COMPUTE = os.getenv("ASR_COMPUTE", "auto")
ASR_BEAM = int(os.getenv("ASR_BEAM", "5"))

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

PIPELINE_NAME = "pyannote/speaker-diarization-3.1"