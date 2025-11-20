import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import LLM_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

def _run_llm(prompt: str, max_new_tokens=400):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
    3. SPEAKER 순서는 입력 회의록의 등장 순서를 유지한다. 
    4. 각 SPEAKER는 반드시 한 번씩만 등장한다. 
    5. 요약문에는 JSON이나 리스트, bullet(-) 기호, 숫자 리스트 등을 절대 사용하지 않는다. 
    6. 오직 아래 형식으로만 출력한다: 
    SPEAKER_00: (핵심 내용) 
    SPEAKER_01: (핵심 내용) 
    SPEAKER_02: (핵심 내용) 
    7. 원본 내용을 기반으로만 요약하고 새로운 정보를 만들어내지 말 것. 
    8. 문장 반복 금지. 

    입력 회의록: 
    {text}
    
    화자별 요약문:
"""
    result = _run_llm(prompt, max_new_tokens=300) 
    
    # 불필요한 텍스트 정리 
    if "화자별 요약문:" in result: 
        result = result.split("화자별 요약문:")[-1].strip() 
        return result.strip()

def extract_tasks(text: str):
    """
    회의록 기반 화자별 Action Items 추출
    JSON 형식으로 반환
    """

    prompt = f"""
    \"\"\" 회의록 기반 화자별 Action Items 추출 JSON 형식으로 반환 \"\"\"

    당신은 회의 분석 전문가입니다.
    아래 회의록은 Whisper + Pyannote 기반 화자 분리 텍스트입니다.

    만약 회의 내에 '해야 할 일', '할당된 업무', '약속된 작업'이 없다면 tasks는 빈 배열([])로 출력하세요.

    각 SPEAKER가 맡은 업무를 찾고, Action Items를 JSON으로 생성하세요.

    규칙:
    - JSON 외의 문장은 절대 출력하지 말 것
    - 할 일이 없으면 빈 배열([])로 출력
    - 각 SPEAKER의 items가 비어있을 수 있음
    - priority는 '높음/보통/낮음'
    - due는 없으면 '미정' 출력

    (JSON 예시):
    {{
    "tasks": [
        {{
        "speaker": "SPEAKER_00",
        "items": [
            {{
            "description": "업무 내용",
            "priority": "보통",
            "due": "미정"
            }}
        ]
        }}
    ]
    }}

    회의 내용:
    {text}

    JSON만 출력하세요.
    """
    
    # LLM 실행
    raw = _run_llm(prompt, max_new_tokens=800)

    # (1) JSON 파싱 시도
    try:
        data = json.loads(raw)
    except Exception:
        return {"tasks": []}

    # (2) tasks 필드 자체가 없으면 빈 리스트
    if "tasks" not in data:
        return {"tasks": []}

    tasks = data["tasks"]

    # (3) 각 화자 items가 모두 빈 배열이면 Action 없음
    if all(len(item.get("items", [])) == 0 for item in tasks):
        return {"tasks": []}

    return data