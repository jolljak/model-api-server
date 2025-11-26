from fastapi import APIRouter, HTTPException, Body
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
        if d["runtimeAuthKey"] and len(d["runtimeAuthKey"]) > 5:
            d["runtimeAuthKey"] = d["runtimeAuthKey"][:5] + "***"
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
    """
    if not payload.items: 
        return {"status": "skipped", "message": "No items to execute"}
    
    logs = await processor.execute_actions(payload.items, tool_manager)
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