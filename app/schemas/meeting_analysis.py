from typing import List, Literal
from pydantic import BaseModel, Field

try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    from langchain.output_parsers import PydanticOutputParser

Priority = Literal["높음", "중간", "낮음"]

class TaskItem(BaseModel):
    업무설명: str = Field(..., description="실행 가능한 업무 한 줄")
    priority: Priority = Field(..., description="높음|중간|낮음")

class SpeakerTasks(BaseModel):
    speaker: str = Field(..., description='예: "S0", "S1"')
    items: List[TaskItem] = Field(default_factory=list)

class MeetingAnalysisResult(BaseModel):
    summary: str = Field(..., description="회의 전체 내용을 한 문단으로 요약")
    tasks: List[SpeakerTasks] = Field(..., description="화자별 업무 리스트")

def get_meeting_analysis_parser():
    return PydanticOutputParser(pydantic_object=MeetingAnalysisResult)
