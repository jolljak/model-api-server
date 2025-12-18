from typing import List, Optional
from pydantic import BaseModel

# 1. 서비스 정의 DTO (DB 및 AI 가이드용)
class ServiceDefinition(BaseModel):
    serviceId: int
    serviceName: str
    actionName: str
    iconPath: Optional[str] = None
    apiUrlTemplate: str
    authType: str
    methodList: str
    payloadTemplate: str
    headerTemplate: Optional[str] = "{}"
    createUserId: str = "admin"
    token: Optional[str] = None 
    
    # [NEW] AI가 알려주는 인증 가이드 (프론트엔드 연동용)
    authGuideUrl: Optional[str] = None
    authGuideHint: Optional[str] = None

# 2. 회의 분석 결과 내부 항목
class ActionItem(BaseModel):
    description: str
    assignee: Optional[str] = None
    priority: str = "Medium"
    due_date: Optional[str] = None
    is_automatable: bool

# 3. 회의 분석 전체 결과
class MeetingAnalysisResult(BaseModel):
    summary: str
    key_decisions: List[str]
    action_items: List[ActionItem]

# 4. API 요청 Body DTO
class AnalyzeRequest(BaseModel):
    transcript: str

class ExecuteRequest(BaseModel):
    items: List[ActionItem]
    serviceType: Optional[str] = None  # Optional: "github", "notion", etc.
    config: Optional[dict] = None      # Optional: { "token": "...", ... }
    context: Optional[dict] = None     # Optional: { "projectName": "...", "userName": "..." }

class ServiceGuideRequest(BaseModel):
    request: str
    
class ServicePresetRequest(BaseModel):
    serviceType: str  # "github", "notion", "jira", "google_calendar"
    config: dict      # { "token": "...", "repo": "...", "database_id": "..." }