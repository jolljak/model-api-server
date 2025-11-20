import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

ASR_MODEL = os.getenv("ASR_MODEL", "base")
ASR_COMPUTE = os.getenv("ASR_COMPUTE", "auto")
ASR_BEAM = int(os.getenv("ASR_BEAM", "5"))

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
