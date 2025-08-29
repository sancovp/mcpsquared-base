"""Request models for phase tools"""

from pydantic import BaseModel
from typing import List


class MCPConfig(BaseModel):
    name: str
    command: str
    args: List[str]
    transport: str


class ToolsFileRequest(BaseModel):
    tools_file_path: str
    user_requirements: str = ""


class DesignsFileRequest(BaseModel):
    designs_file_path: str


