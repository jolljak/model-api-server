from langchain_ollama import ChatOllama
from app.core.config import OLLAMA_MODEL

_llm = None

def get_llm_model():
    """
    Ollama기반 로컬 LLM (Qwen 등) 싱글톤 반환
    """
    global _llm
    if _llm is None:
        try:
            # temperature=0: 분석/추출 작업이므로 창의성보다는 정확도/일관성 중시
            _llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
            print(f"[INFO] Local LLM ({OLLAMA_MODEL}) Loaded.")
        except Exception as e:
            print(f"[WARN] Failed to load LLM: {e}")
            return None
    return _llm