"""
Pydantic models for workflow and agent configuration schemas
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class TemplatedArg(BaseModel):
    """Schema for a templated argument"""
    name: str = Field(..., description="Name of the template variable (e.g., 'user_param')")
    description: str = Field(..., description="Human-readable description of what this arg should contain")
    type: str = Field(default="string", description="Expected type: string, number, boolean, object")
    required: bool = Field(default=True, description="Whether this argument is required")
    default: Optional[Union[str, int, bool, Dict[str, Any]]] = Field(None, description="Default value if not provided")


class MCPServerConfig(BaseModel):
    """Schema for MCP server configuration"""
    command: str = Field(..., description="Command to run the MCP server")
    args: List[str] = Field(default_factory=list, description="Command line arguments")
    transport: str = Field(default="stdio", description="Transport protocol (stdio, websocket)")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    cwd: Optional[str] = Field(None, description="Working directory")


class MCPClientConfig(BaseModel):
    """Schema for MCP client configuration"""
    mcpServers: Dict[str, MCPServerConfig] = Field(..., description="Map of server names to configurations")


class WorkflowConfig(BaseModel):
    """Schema for workflow configuration"""
    workflow_name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Clear description of what the workflow does")
    agent_config_name: str = Field(..., description="Name of the agent config to use")
    input_prompt: str = Field(..., description="Template prompt with {{placeholders}}")
    templated_args: List[TemplatedArg] = Field(..., description="Structured template arguments with descriptions")
    tool_sequence: List[str] = Field(..., description="Ordered list of tools to execute")
    domain: Optional[str] = Field(None, description="Domain category (e.g., 'knowledge_management', 'file_operations')")


class AgentConfig(BaseModel):
    """Schema for agent configuration - supports both mcp_names and mcp_config formats"""
    agent_name: str = Field(..., description="Name of the agent")
    mcp_names: Optional[List[str]] = Field(None, description="List of MCP names for agent loader to resolve")
    mcp_config: Optional[MCPClientConfig] = Field(None, description="Direct MCP client configuration")
    system_prompt: str = Field(..., description="System prompt for the agent")
    allowed_tools: List[str] = Field(..., description="List of allowed prefixed tool names")
    model: str = Field(default="gpt-5-mini", description="LLM model to use")
    provider: str = Field(default="openai", description="LLM provider")
    max_steps: int = Field(default=20, description="Maximum steps for agent execution")
    
    def model_post_init(self, __context):
        """Validate that either mcp_names or mcp_config is provided"""
        if not self.mcp_names and not self.mcp_config:
            raise ValueError("Either mcp_names or mcp_config must be provided")
        if self.mcp_names and self.mcp_config:
            raise ValueError("Provide either mcp_names OR mcp_config, not both")


class WorkflowDesigns(BaseModel):
    """Schema for complete workflow designs file"""
    workflows: List[WorkflowConfig] = Field(..., description="List of workflow configurations")
    mcp_name: str = Field(..., description="Name of the source MCP")
    generated_at: str = Field(..., description="Timestamp of generation")
    
    def to_json_list(self) -> List[Dict[str, Any]]:
        """Convert to list format for JSON file output"""
        return [workflow.dict() for workflow in self.workflows]


class AgentConfigs(BaseModel):
    """Schema for agent configurations collection"""
    agents: List[AgentConfig] = Field(..., description="List of agent configurations")
    mcp_name: str = Field(..., description="Name of the source MCP")
    generated_at: str = Field(..., description="Timestamp of generation")
    
    def to_agent_files(self) -> Dict[str, Dict[str, Any]]:
        """Convert to individual agent config files"""
        return {f"{agent.agent_name}.json": agent.dict() for agent in self.agents}