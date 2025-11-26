import os
import re
import json
import httpx
from typing import List, Type
from pydantic import BaseModel, Field, create_model
from app.schemas.mcp import ServiceDefinition, ServicePresetRequest
try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool

from app.schemas.mcp import ServiceDefinition
from app.core.config import SLACK_BOT_TOKEN

class DynamicToolManager:
    def __init__(self):
        self.tools_cache: List[StructuredTool] = []
        self.temp_registered_services: List[ServiceDefinition] = []

    def _extract_variables(self, template: str) -> List[str]:
        # 템플릿 문자열에서 {variable} 추출
        return re.findall(r"\{([a-zA-Z0-9_]+)\}", template)

    def _create_pydantic_model_from_vars(self, tool_name: str, variables: List[str]) -> Type[BaseModel]:
        # LangChain Tool용 입력 스키마 동적 생성
        fields = {var: (str, Field(..., description=f"The value for {var}")) for var in variables}
        return create_model(f"{tool_name}Input", **fields)

    async def register_temp_service(self, service: ServiceDefinition):
        # 메모리에 서비스 임시 등록 (중복 방지)
        self.temp_registered_services = [
            s for s in self.temp_registered_services 
            if not (s.serviceId == service.serviceId and s.actionName == service.actionName)
        ]
        self.temp_registered_services.append(service)
        # 도구 목록 갱신
        await self.refresh_tools()

    async def get_all_services(self) -> List[ServiceDefinition]:
        services = []
        
        # 1. 기본 내장 서비스 (Slack)
        if SLACK_BOT_TOKEN:
            services.append(ServiceDefinition(
                serviceId=999,
                serviceName="Slack",
                actionName="send_message",
                apiUrlTemplate="https://slack.com/api/chat.postMessage",
                authType="bearer",
                methodList="POST",
                payloadTemplate='{"channel": "{channel}", "text": "{text}"}',
                headerTemplate='{}',
                runtimeAuthKey=SLACK_BOT_TOKEN
            ))
            
        # 2. 동적으로 등록된 서비스 병합
        services.extend(self.temp_registered_services)
        return services

    async def refresh_tools(self):
        services = await self.get_all_services()
        new_tools = []

        for svc in services:
            try:
                unique_tool_name = f"{svc.serviceName}_{svc.actionName}"
                
                # 변수 추출 (URL, Body, Header 전체)
                vars_from_url = self._extract_variables(svc.apiUrlTemplate)
                vars_from_body = self._extract_variables(svc.payloadTemplate)
                vars_from_header = self._extract_variables(svc.headerTemplate or "")
                needed_vars = list(set(vars_from_url + vars_from_body + vars_from_header))

                input_model = self._create_pydantic_model_from_vars(unique_tool_name, needed_vars)

                # --- 실행 함수 (Closure) ---
                async def _executor(_svc=svc, **kwargs):
                    print(f"\n[DEBUG] 🛠️ Executing Tool: {_svc.serviceName}:{_svc.actionName}")
                    
                    final_url = _svc.apiUrlTemplate
                    final_body = _svc.payloadTemplate
                    final_header_str = _svc.headerTemplate or "{}"
                    
                    try:
                        # 변수 치환
                        for k, v in kwargs.items():
                            val = str(v).replace('"', '\\"')
                            if f"{{{k}}}" in final_url: 
                                final_url = final_url.replace(f"{{{k}}}", str(v))
                            if f"{{{k}}}" in final_body: 
                                final_body = final_body.replace(f"{{{k}}}", val)
                            if f"{{{k}}}" in final_header_str:
                                final_header_str = final_header_str.replace(f"{{{k}}}", val)
                        
                        # JSON 파싱
                        json_body = None
                        if final_body and final_body.strip() != "{}":
                            json_body = json.loads(final_body)

                        custom_headers = {}
                        if final_header_str.strip() != "{}":
                            custom_headers = json.loads(final_header_str)

                    except Exception as e:
                        return f"Error processing parameters: {str(e)}"

                    # 헤더 설정
                    headers = {"Content-Type": "application/json; charset=utf-8"}
                    headers.update(custom_headers)

                    # 인증 토큰 주입
                    token = _svc.runtimeAuthKey
                    if token:
                        if _svc.authType == "bearer": 
                            headers["Authorization"] = f"Bearer {token}"
                        elif _svc.authType == "api-key": 
                            headers["x-api-key"] = token
                    
                    print(f"[DEBUG] 📡 Sending {_svc.methodList} to {final_url}")
                    
                    # API 요청 전송
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            method = _svc.methodList.upper()
                            if method == "POST":
                                resp = await client.post(final_url, json=json_body, headers=headers)
                            elif method == "GET":
                                resp = await client.get(final_url, params=json_body, headers=headers)
                            elif method == "PATCH":
                                resp = await client.patch(final_url, json=json_body, headers=headers)
                            elif method == "PUT":
                                resp = await client.put(final_url, json=json_body, headers=headers)
                            elif method == "DELETE":
                                resp = await client.delete(final_url, headers=headers)
                            else:
                                return f"Method {method} not supported"
                            
                            if resp.status_code < 300:
                                return f"Success: {resp.text[:2000]}" # 너무 긴 응답은 자름
                            else:
                                return f"Failed (HTTP {resp.status_code}): {resp.text}"
                    except Exception as req_err:
                        return f"Network Error: {str(req_err)}"

                # LangChain Tool 객체 생성
                desc = f"Tool to perform '{svc.actionName}' on '{svc.serviceName}'. Required inputs: {needed_vars}"
                tool = StructuredTool.from_function(
                    coroutine=_executor,
                    name=unique_tool_name,
                    description=desc,
                    args_schema=input_model
                )
                new_tools.append(tool)

            except Exception as e:
                print(f"[ERROR] Tool Registration Failed ({svc.serviceName}): {e}")

        self.tools_cache = new_tools
        return self.tools_cache
    async def register_preset(self, request: ServicePresetRequest) -> ServiceDefinition:
        stype = request.serviceType.lower()
        cfg = request.config
        
        definition = None
        
        # 1. GitHub
        if stype == "github":
            owner = cfg.get("owner")
            repo = cfg.get("repo")
            token = cfg.get("token")
            
            definition = ServiceDefinition(
                serviceId=int(f"101{len(self.temp_registered_services)}"),
                serviceName="GitHub",
                actionName="create_issue",
                apiUrlTemplate=f"https://api.github.com/repos/{owner}/{repo}/issues",
                authType="bearer",
                methodList="POST",
                # GitHub 이슈 생성 표준 페이로드
                payloadTemplate=json.dumps({
                    "title": "{title}",
                    "body": "{body}\n\n> Auto-generated by Mina AI",
                    "labels": ["{label}"] # 예: bug, enhancement
                }),
                headerTemplate='{"Accept": "application/vnd.github+json"}',
                runtimeAuthKey=token,
                authGuideUrl="https://github.com/settings/tokens"
            )

        # 2. Notion
        elif stype == "notion":
            db_id = cfg.get("database_id")
            token = cfg.get("token")
            
            definition = ServiceDefinition(
                serviceId=int(f"201{len(self.temp_registered_services)}"),
                serviceName="Notion",
                actionName="create_page",
                apiUrlTemplate="https://api.notion.com/v1/pages",
                authType="bearer",
                methodList="POST",
                # 노션 페이지 생성 표준 페이로드
                payloadTemplate=json.dumps({
                    "parent": { "database_id": "{database_id}" },
                    "properties": {
                        "Name": { "title": [ { "text": { "content": "{title}" } } ] },
                        "Tags": { "multi_select": [ { "name": "Meeting Note" } ] }
                    },
                    "children": [
                        {
                            "object": "block",
                            "type": "heading_2",
                            "heading_2": { "rich_text": [{ "text": { "content": "📌 요약" } }] }
                        },
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": { "rich_text": [{ "text": { "content": "{content}" } }] }
                        }
                    ]
                }),
                headerTemplate='{"Notion-Version": "2022-06-28"}',
                runtimeAuthKey=token,
                authGuideUrl="https://www.notion.so/my-integrations"
            )

        # 3. Jira
        elif stype == "jira":
            domain = cfg.get("domain") # sub-domain only
            email = cfg.get("email")
            token = cfg.get("token")
            project_key = cfg.get("project_key")
            
            # Jira는 Basic Auth (email:token)를 Base64로 써야 하지만, 여기선 편의상 Bearer 로직을 재활용하거나
            # tool_manager 실행 시 header 커스텀이 필요함. 일단은 구조만 잡음.
            definition = ServiceDefinition(
                serviceId=int(f"301{len(self.temp_registered_services)}"),
                serviceName="Jira",
                actionName="create_ticket",
                apiUrlTemplate=f"https://{domain}.atlassian.net/rest/api/3/issue",
                authType="api-key", # 별도 처리 필요 가능
                methodList="POST",
                payloadTemplate=json.dumps({
                    "fields": {
                        "project": { "key": project_key },
                        "summary": "{summary}",
                        "description": "{description}",
                        "issuetype": { "name": "Task" }
                    }
                }),
                runtimeAuthKey=token, # 실제론 email:token 인코딩 필요
                authGuideUrl="https://id.atlassian.com/manage-profile/security/api-tokens"
            )

        # 4. Google Calendar
        elif stype == "google_calendar":
            token = cfg.get("token") # Access Token
            calendar_id = cfg.get("calendar_id", "primary")
            
            definition = ServiceDefinition(
                serviceId=int(f"401{len(self.temp_registered_services)}"),
                serviceName="GoogleCalendar",
                actionName="add_event",
                apiUrlTemplate=f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
                authType="bearer",
                methodList="POST",
                payloadTemplate=json.dumps({
                    "summary": "{summary}",  
                    "description": "Mina AI에서 자동 생성된 업무입니다. (우선순위: {priority})",
                    "start": { "date": "{due_date}" }, 
                    "end": { "date": "{due_date}" }
                }),
                runtimeAuthKey=token,
                authGuideUrl="https://developers.google.com/oauthplayground"
            )
            
        if definition:
            await self.register_temp_service(definition)
            return definition
        else:
            raise ValueError("Unknown Service Type")