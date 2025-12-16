from typing import List

# LangChain 관련
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    create_react_agent = None

# 내부 모듈
from app.schemas.mcp import MeetingAnalysisResult, ActionItem, ServiceDefinition
from app.schemas.meeting_analysis import get_meeting_analysis_parser
from app.services.tool_manager import DynamicToolManager
from app.models.llm_model import get_llm_model # [중요] 싱글톤 LLM 로더

class MeetingProcessor:
    def __init__(self):
        self.parser = get_meeting_analysis_parser()
        self.service_parser = PydanticOutputParser(pydantic_object=ServiceDefinition)

    async def analyze_transcript(self, transcript: str) -> MeetingAnalysisResult:
        """회의록 텍스트를 분석하여 요약 및 Action Item 추출"""
        llm = get_llm_model()
        if not llm: raise ValueError("LLM not loaded")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "너는 회의 텍스트를 분석하는 도우미야.\n"
            "반드시 아래 스키마에 맞는 JSON만 출력해.\n"
            "{format_instructions}\n"
            "규칙:\n"
            "- 입력은 화자(S0, S1, ...)별 발언이 섞여 있는 회의 로그다.\n"
            "- summary는 화자 구분 없이 회의 전체를 하나의 문단으로 요약한다.\n"
            "- 회의와 무관한 잡담은 요약에서 최소화하거나 제외한다.\n"
            "- tasks는 화자별로 실제 실행 가능한 업무만 추출한다.\n"
            "- 업무가 없는 화자는 items를 빈 배열로 둔다.\n"
            "- priority는 높음/중간/낮음 중 하나만 사용한다.\n"
            ),
            ("human", "{transcript}")
        ])
        
        chain = prompt | llm | self.parser
        return await chain.ainvoke({
            "transcript": transcript,
            "format_instructions": self.parser.get_format_instructions()
        })

    async def execute_actions(self, actions: List[ActionItem], tool_manager: DynamicToolManager):
        """Action Item 목록을 받아 적절한 도구를 실행"""
        llm = get_llm_model()
        if not llm: return [{"error": "LLM not loaded"}]
        if not create_react_agent: return [{"error": "LangGraph not installed"}]
        
        tools = await tool_manager.refresh_tools()
        if not tools: return [{"error": "No tools available"}]

        system_prompt = (
            "You are an AI assistant. You MUST use the provided tools to execute requests. "
            "Do NOT just reply with text. Call the tool function directly."
        )
        
        graph = create_react_agent(llm, tools=tools)
        logs = []
        
        for action in actions:
            print(f"[*] Processing Action: {action.description}")
            try:
                inputs = {"messages": [("system", system_prompt), ("user", f"TASK: {action.description}")]}
                result = await graph.ainvoke(inputs)
                
                last_msg = result["messages"][-1]
                tool_calls = [m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls]
                
                status = "success" if tool_calls else "failed"
                logs.append({
                    "task": action.description, 
                    "status": status, 
                    "ai_response": str(last_msg.content)
                })
            except Exception as e:
                logs.append({"task": action.description, "status": "error", "error": str(e)})
        return logs

    async def generate_service_definition(self, user_request: str) -> ServiceDefinition:
        """사용자의 자연어 요청을 분석하여 서비스 등록 설정(JSON) 생성"""
        llm = get_llm_model()
        if not llm: raise ValueError("LLM not loaded")
        
        system_prompt = """
        You are an API Integration Expert. Your goal is to configure 'Universal MCP Client' for the user.
        
        --- KNOWN SERVICE TEMPLATES (Prioritize these) ---
        
        1. [Notion] (Meeting Notes)
           - Goal: Create a page in a database.
           - API: POST https://api.notion.com/v1/pages
           - Headers: {{"Notion-Version": "2022-06-28"}}
           - Auth Hint: "Need 'Secret Key' and 'Database ID'."
           - Auth URL: https://www.notion.so/my-integrations
           - Template Vars: {{database_id}}, {{meeting_title}}, {{summary_content}}
        
        2. [Google Calendar] (Deadlines)
           - Goal: Add an event.
           - API: POST https://www.googleapis.com/calendar/v3/calendars/primary/events
           - Auth Hint: "Need OAuth Access Token."
           - Auth URL: https://developers.google.com/oauthplayground/
           - Template Vars: {{task_name}}, {{due_date}}, {{priority}}
        
        3. [GitHub] (Issues)
           - Goal: Create an issue.
           - API: POST https://api.github.com/repos/{{owner}}/{{repo}}/issues
           - Headers: {{"Accept": "application/vnd.github+json"}}
           - Auth Hint: "Need PAT(Classic) with 'repo' scope."
           - Auth URL: https://github.com/settings/tokens
           - Template Vars: {{issue_title}}, {{issue_description}}, {{label}}
        
        4. [Jira] (Tasks)
           - Goal: Create an issue (Task).
           - API: POST https://{{domain}}.atlassian.net/rest/api/3/issue
           - Auth Hint: "Need API Token and Project Key (e.g., KAN)."
           - Auth URL: https://id.atlassian.com/manage-profile/security/api-tokens
           - Template Vars: {{project_key}}, {{task_summary}}, {{task_detail}}
        
        --- INSTRUCTIONS ---
        1. Analyze the user's request.
        2. Select the matching template.
        3. Construct the JSON schema (apiUrlTemplate, payloadTemplate, authGuideUrl, etc.).
        4. If unknown, generate a generic schema.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n{format_instructions}"),
            ("human", "{request}")
        ])
        
        chain = prompt | llm | self.service_parser
        return await chain.ainvoke({
            "request": user_request,
            "format_instructions": self.service_parser.get_format_instructions()
        })