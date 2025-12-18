from fastapi import APIRouter, HTTPException, Body, Header
import httpx
from typing import List, Dict, Any

from app.schemas.mcp import (
    ServiceDefinition, ServiceGuideRequest, 
    AnalyzeRequest, ExecuteRequest
)

from app.schemas.mcp import ServicePresetRequest
from app.services.tool_manager import DynamicToolManager
from app.services.processor import MeetingProcessor

router = APIRouter()

tool_manager = DynamicToolManager()
processor = MeetingProcessor()

@router.post("/services/guide")
async def service_guide(payload: ServiceGuideRequest):
    """
    [AI 가이드] 사용자의 자연어 요청("노션 연결해줘")을 받아 
    서비스 등록에 필요한 설정값(JSON)을 자동 생성해줍니다.
    """
    try:
        definition = await processor.generate_service_definition(payload.request)
        return {"status": "success", "suggested_config": definition.dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/services/register")
async def register_service(payload: ServiceDefinition):
    """
    새로운 MCP 서비스(도구)를 메모리에 임시 등록합니다.
    """
    await tool_manager.register_temp_service(payload)
    return {"status": "registered", "service": f"{payload.serviceName}:{payload.actionName}"}

@router.get("/services/list")
async def list_services():
    """
    현재 등록된 모든 서비스 목록을 반환합니다.
    (보안을 위해 키 값은 마스킹 처리됨)
    """
    services = await tool_manager.get_all_services()
    sanitized = []
    for svc in services:
        d = svc.dict()
        if d["token"] and len(d["token"]) > 5:
            d["token"] = d["token"][:5] + "***"
        sanitized.append(d)
    return {"status": "success", "count": len(sanitized), "services": sanitized}

@router.post("/analyze-meeting")
async def analyze_meeting(payload: AnalyzeRequest):
    """
    회의록 텍스트를 AI가 분석하여 요약 및 할 일(Action Item)을 추출합니다.
    """
    try:
        result = await processor.analyze_transcript(payload.transcript)
        return {"status": "success", "analysis": result.dict()}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/execute-actions")
async def execute_actions(payload: ExecuteRequest):
    """
    추출된 할 일(Action Item)을 실제 연결된 도구(Jira, GitHub 등)로 실행합니다.
    serviceType과 config가 포함된 경우 자동으로 도구를 선등록(Warm-up)하고 실행합니다.
    """
    if not payload.items: 
        return {"status": "skipped", "message": "No items to execute"}
    
    # [NEW] 실행 요청 시 서비스 정보가 포함되어 있으면 자동 등록 수행
    service_def = None
    if payload.serviceType and payload.config:
        try:
            from app.schemas.mcp import ServicePresetRequest
            preset_payload = ServicePresetRequest(
                serviceType=payload.serviceType,
                config=payload.config
            )
            service_def = await tool_manager.register_preset(preset_payload)
            print(f"[INFO] 🔄 Auto-registered service '{payload.serviceType}' from execute-actions payload.")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to auto-register service: {str(e)}")

    logs = await processor.execute_actions(payload.items, tool_manager, payload.context, service_def)
    return {"status": "completed", "logs": logs}

@router.post("/services/register-preset")
async def register_preset(payload: ServicePresetRequest):
    """
    [간편 등록] 프론트엔드에서 서비스 타입과 필수 키값만 받아서 등록합니다.
    """
    try:
        service = await tool_manager.register_preset(payload)
        return {"status": "success", "service": service.dict()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/services/google-calendar/events")
async def get_google_calendar_events(authorization: str = Header(None)):
    """
    구글 캘린더의 최근 일정 5개를 조회합니다.
    Authorization 헤더에 'Bearer {access_token}' 형식이 필요합니다.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.split(" ")[1]
    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    params = {
        "maxResults": 5,
        "orderBy": "startTime",
        "singleEvents": True,
        "timeMin": "2024-01-01T00:00:00Z" # 실제로는 현재 시간 기준으로 필터링 가능
    }
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code != 200:
                return {"status": "error", "code": resp.status_code, "message": resp.text}
            
            data = resp.json()
            events = []
            for item in data.get("items", []):
                events.append({
                    "id": item.get("id"),
                    "summary": item.get("summary"),
                    "start": item.get("start", {}).get("dateTime") or item.get("start", {}).get("date"),
                    "end": item.get("end", {}).get("dateTime") or item.get("end", {}).get("date"),
                    "location": item.get("location")
                })
            return {"status": "success", "events": events}
    except Exception as e:
        return {"status": "error", "message": str(e)}