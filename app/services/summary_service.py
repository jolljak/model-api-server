import json
import torch
from app.core.config import LLM_MODEL

# 전역 변수로 모델 캐싱 (처음엔 None으로 비워둠)
_tokenizer = None
_model = None

def get_legacy_llm():
    """
    기존 HuggingFace Transformers 모델 로드
    """
    global _tokenizer, _model
    
    # 이미 로드되어 있으면 반환
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    print("[INFO] Loading Legacy LLM (Transformers)... This may take memory.")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("[INFO] Legacy LLM Loaded.")
        return _tokenizer, _model
    except ImportError:
        print("[ERROR] 'transformers' library not found. Please install it.")
        return None, None
    except Exception as e:
        print(f"[ERROR] Legacy LLM Load Failed: {e}")
        return None, None

def _run_llm(prompt: str, max_new_tokens=400):
    tokenizer, model = get_legacy_llm()
    if not model:
        return "Error: Legacy Model not loaded."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def summarize_text(text: str):
    prompt = f""" 
    당신은 회의 내용을 '화자 단위'로 핵심만 요약하는 전문가입니다. 
    
    입력은 다음 형식입니다: [SSTART~END] SPEAKER_ID: 발언 내용 
    요약 규칙: 1. 시간 정보([0.3~1.2] 등)는 모두 제거한다. 
    2. 같은 SPEAKER가 여러 번 말한 경우, 전체 발언의 핵심만 한 문단으로 압축하여 요약한다. 
    3. SPEAKER 순서대로 출력한다.
    
    [회의록]:
    {text}
    
    화자별 요약문:
    """
    result = _run_llm(prompt, max_new_tokens=300)
    
    if "화자별 요약문:" in result: 
        result = result.split("화자별 요약문:")[-1].strip() 
    return result.strip()

def extract_tasks(text: str):
    prompt = f"""
    \"\"\" 회의록 기반 화자별 Action Items 추출 JSON 형식으로 반환 \"\"\"

    당신은 회의 분석 전문가입니다.
    아래 회의록은 Whisper + Pyannote 기반 화자 분리 텍스트입니다.
    JSON 외의 문장은 절대 출력하지 마세요.

    (JSON 예시):
    {{
    "tasks": [
        {{ "speaker": "SPEAKER_00", "items": [ {{ "description": "...", "priority": "높음" }} ] }}
    ]
    }}

    [회의록]:
    {text}
    """
    # 기존 로직 유지
    return _run_llm(prompt, max_new_tokens=500)