"""
MCPSquaredAgent - Core agent for workflow generation using mcp-use framework

This agent coordinates the 7-phase workflow generation process:
Phase 1: MCP Analysis (install, discover tools)
Phase 2: Workflow Design (design workflows, create agent configs)  
Phase 3: Package Generation (render implementations, documentation, packaging)
"""

from typing import Dict, Any, Optional
from mcp_use import MCPClient, MCPAgent
from langchain_openai import ChatOpenAI
import json
import tempfile
import os
import logging
import traceback
from utils.workflow_runner import run_workflow, _load_workflow_config
from utils.agent_loader import _load_json_config
from tools.chain_selection_mcp.tools.chain_selection import (
    get_last_chain_file, 
    format_chain_for_executor,
    read_chain_selection,
    get_most_recent_chain_file
)

# Setup logging
logger = logging.getLogger(__name__)


class MCPSquaredAgent:
    """
    MCPSquaredAgent uses mcp-use MCPAgent with phase tools to generate workflow packages.
    
    This follows the exact patterns from heaven_mcp_use_sidecar container.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MCPSquaredAgent with configuration.
        
        Args:
            config: Agent configuration containing model, system_prompt, tools, etc.
                   Must include selector_workflow and executor_workflow paths.
        """
        self.config = config
        self.client = None
        self.agent = None
        
        # Selector/Executor workflow paths - core of new architecture
        self.selector_workflow = config.get("selector_workflow", "chain_selector_workflow")
        self.executor_workflow = config.get("executor_workflow", "mcpsquared_executor_workflow")
        
        logger.info(f"Initializing MCPSquaredAgent with config: {config.get('name', 'unknown')}")
        logger.debug(f"Selector workflow: {self.selector_workflow}")
        logger.debug(f"Executor workflow: {self.executor_workflow}")
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup MCPClient and MCPAgent following mcp-use patterns"""
        logger.debug("Setting up MCPClient and MCPAgent")
        
        # Create MCPClient with phase tools server
        phase_tools_config = self._get_phase_tools_config()
        client_config = {
            "mcpServers": {
                "mcpsquared_phase_tools": phase_tools_config
            }
        }
        
        self.client = MCPClient.from_dict(client_config)
        logger.debug("MCPClient created successfully")
        
        # Create MCPAgent with LangChain integration (following sidecar patterns)
        # ChatOpenAI will get OPENAI_API_KEY from environment automatically
        self.agent = MCPAgent(
            llm=ChatOpenAI(model=self.config.get("model", "gpt-5-mini")),
            client=self.client,
            use_server_manager=False,  # Use specific server, not auto-select
            system_prompt=self.config.get("system_prompt", self._get_default_system_prompt()),
            max_steps=self.config.get("max_steps", 20)
        )
        logger.info("MCPAgent setup completed successfully")
    
    def _get_mcpsquared_paths(self) -> tuple:
        """Get absolute paths for MCPSquared directories."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcpsquared_dir = os.path.dirname(current_dir)
        phase_tools_path = os.path.join(mcpsquared_dir, "tools", "phase_tools_server.py")
        return current_dir, mcpsquared_dir, phase_tools_path
    
    def _build_phase_tools_base_config(self, mcpsquared_dir: str) -> Dict[str, Any]:
        """Build base configuration for phase tools server."""
        return {
            "command": "python",
            "args": ["-m", "tools.mcpsquared_workflow_mcp.phase_tools_server"],
            "transport": "stdio",
            "cwd": mcpsquared_dir
        }
    
    def _get_provider_key_mapping(self) -> Dict[str, str]:
        """Get mapping of providers to environment variable names."""
        return {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY"
        }
    
    def _validate_and_get_provider(self) -> str:
        """Validate provider setting and return normalized provider name."""
        provider = self.config.get("provider", "openai").lower()
        provider_key_map = self._get_provider_key_mapping()
        
        if provider not in provider_key_map:
            logger.warning(f"Unknown provider '{provider}', defaulting to openai")
            return "openai"
        return provider
    
    def _add_provider_key_to_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add correct API key to config based on provider setting."""
        provider = self._validate_and_get_provider()
        provider_key_map = self._get_provider_key_mapping()
        
        env_var_name = provider_key_map[provider]
        api_key = os.getenv(env_var_name)
        
        if api_key:
            config["env"] = {env_var_name: api_key}
            logger.debug(f"Added {env_var_name} to phase tools config")
        else:
            logger.warning(f"No {env_var_name} found in environment - phase tools LLM subagents may fail")
        
        return config

    def _get_phase_tools_config(self) -> Dict[str, Any]:
        """
        Get MCP server configuration for phase tools.
        
        Returns:
            MCP server config for phase tools server
        """
        _, mcpsquared_dir, _ = self._get_mcpsquared_paths()
        config = self._build_phase_tools_base_config(mcpsquared_dir)
        return self._add_provider_key_to_config(config)
    
    async def get_available_tools_from_phase_server(self) -> list:
        """
        Get list of available tools from the phase server.
        
        Returns:
            List of available tool names from phase server
        """
        logger.info("Getting available tools from phase server")
        
        try:
            # Ensure client sessions are created
            if self.client:
                await self.client.create_all_sessions()
                logger.debug(f"Created sessions: {list(self.client.sessions.keys())}")
                
                # Get tools from the phase server session
                if "mcpsquared_phase_tools" in self.client.sessions:
                    session = self.client.sessions["mcpsquared_phase_tools"]
                    tools_list = await session.list_tools()
                    tool_names = [tool.name for tool in tools_list]
                    logger.info(f"Found {len(tool_names)} tools in phase server: {tool_names}")
                    return tool_names
                else:
                    logger.error("Phase server session 'mcpsquared_phase_tools' not found")
                    return []
            else:
                logger.error("No MCP client available")
                return []
                
        except Exception as e:
            logger.error(f"Error getting tools from phase server: {str(e)}")
            return []

    async def call_selector(self, user_prompt: str, available_tools: list) -> list:
        """
        Call selector workflow to choose tool sequence.
        
        Args:
            user_prompt: User's request string
            available_tools: List of available tools from phase server
            
        Returns:
            List of selected tools from chain selection file
        """
        logger.info(f"Calling selector workflow: {self.selector_workflow}")
        
        selector_request = {
            "user_request": user_prompt,
            "available_tools": available_tools
        }
        
        await run_workflow(self.selector_workflow, selector_request)
        
        # Get the most recent chain file (ignore agent name since LLM might make up its own)
        chain_file = get_most_recent_chain_file()
        logger.debug(f"Retrieved chain file: {chain_file}")
        
        # Read the selected chain to get the tool sequence
        chain_data = read_chain_selection(chain_file)
        selected_tools = chain_data.get("chain_sequence", [])
        logger.info(f"Selected tool sequence: {selected_tools}")
        
        return selected_tools
    
    async def call_executor(self, user_prompt: str, selected_tools: list, **templated_args) -> Dict[str, Any]:
        """
        Call executor workflow with selected tools as allowed_tools.
        
        Args:
            user_prompt: Original user request string
            selected_tools: List of tools selected by selector
            **templated_args: Additional templated arguments for executor workflow
            
        Returns:
            Result from executor workflow
        """
        # Format selected chain into executor prompt  
        chain_file = get_most_recent_chain_file()
        
        # If we have templated_args, build request dict, otherwise use original string format
        if templated_args:
            request_data = {
                "user_prompt": user_prompt,
                **templated_args  # Include all domain-specific templated args
            }
            enhanced_prompt = format_chain_for_executor(request_data, chain_file)
        else:
            enhanced_prompt = format_chain_for_executor(user_prompt, chain_file)
        
        # Build executor request - pack all templated_args into request string
        request_string = user_prompt
        for key, value in templated_args.items():
            request_string += f"\n{key}: {json.dumps(value) if isinstance(value, dict) else value}"
        
        executor_request = {
            "request": request_string,
            "chain_guidance": enhanced_prompt
        }
        
        # Call executor workflow with selected tools as allowed_tools
        logger.info(f"Calling executor workflow: {self.executor_workflow} with {len(selected_tools)} allowed tools")
        result = await run_workflow(
            self.executor_workflow, 
            executor_request,
            allowed_tools=selected_tools
        )
        
        return result

    async def run(self, user_prompt: str, **templated_args) -> Dict[str, Any]:
        """
        Main entry point for MCPSquaredAgent using selector/executor pattern.
        
        Args:
            user_prompt: Simple user request string
            **templated_args: Additional templated arguments to pass to executor workflow
            
        Returns:
            Result from executor workflow
        """
        logger.info("Starting MCPSquaredAgent run with selector/executor pattern")
        logger.info(f"User prompt: {user_prompt}")
        
        try:
            # 1. Get available tools from phase server
            available_tools = await self.get_available_tools_from_phase_server()
            logger.info(f"Found {len(available_tools)} available tools from phase server")
            
            # 2. Call selector to choose tool sequence
            selected_tools = await self.call_selector(user_prompt, available_tools)
            
            # 3. Call executor with selected tools and templated args
            result = await self.call_executor(user_prompt, selected_tools, **templated_args)
            
            logger.info("MCPSquaredAgent run completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in MCPSquaredAgent run: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "user_prompt": user_prompt
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for MCPSquaredAgent"""
        return """You are an MCP workflow package generator. You MUST call tools sequentially to generate installable packages.

PHASE 1: MCP Analysis
1. CALL phase1_1_install_mcp_tool with the provided MCP config
2. CALL phase1_2_list_mcp_tools_tool with the MCP config to get tools file path

PHASE 2: Workflow Design  
3. CALL phase2_1_call_workflow_designer_subagent_tool with tools file path to get designs file path
4. CALL phase2_2_call_agent_designer_subagent_tool with designs file path to get configs directory

CRITICAL: Actually CALL each tool and use the returned file paths for subsequent tools. Do not just describe - EXECUTE the tools.

Return ONLY the project directory path when complete - nothing else."""
    
    async def generate_workflows_with_selector(self, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate workflow package using the new selector/executor pattern.
        
        This is the domain-specific method that packages MCP config data
        into templated args and calls the generic run() method.
        
        Args:
            mcp_config: User's MCP configuration with name, command, args, transport
            
        Returns:
            Result dictionary with package info or error details
        """
        logger.info(f"Generating workflows for MCP: {mcp_config.get('name', 'unknown')} using selector/executor")
        
        # Build user prompt that describes the task
        user_prompt = f"Generate workflows for the {mcp_config.get('name', 'unknown')} MCP"
        
        # Package MCP config into templated args that executor workflow expects
        # The executor workflow config has templated_args that need these values
        return await self.run(
            user_prompt=user_prompt,
            # These become templated_args that flow to the executor
            mcp_config=mcp_config,
            name=mcp_config.get("name"),
            command=mcp_config.get("command"),
            args=mcp_config.get("args"),
            transport=mcp_config.get("transport")
        )
    
    async def generate_workflow_package(self, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for generating workflow packages.
        
        Args:
            mcp_config: User's MCP configuration to analyze and generate workflows from
            
        Returns:
            Result dictionary with package info or error details
        """
        logger.info(f"Starting workflow package generation for MCP: {mcp_config.get('name', 'unknown')}")
        
        try:
            # Execute MCPSquared workflow using workflow runner
            logger.info("Executing mcpsquared_generation_workflow")
            # Flatten mcp_config to match workflow template expectations
            workflow_args = {
                "name": mcp_config.get("name"),
                "command": mcp_config.get("command"), 
                "args": mcp_config.get("args"),
                "transport": mcp_config.get("transport")
            }
            result = await run_workflow("mcpsquared_generation_workflow", workflow_args)
            logger.info("MCPSquared workflow execution completed")
            
            # Return the actual agent result without parsing
            return {
                "status": "success",
                "result": result,  # The actual agent output - filepath or whatever it returned
                "mcp_config": mcp_config
            }
            
        except Exception as e:
            logger.error(f"Error in generate_workflow_package: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "mcp_config": mcp_config
            }
    
    def _build_generation_prompt(self, mcp_config: Dict[str, Any]) -> str:
        """
        Build the execution prompt for workflow generation.
        
        Args:
            mcp_config: User's MCP configuration
            
        Returns:
            Formatted prompt for MCPAgent execution
        """
        logger.debug("Building generation prompt")
        return f"""Generate workflow MCP package for MCP config: {json.dumps(mcp_config, indent=2)}

Execute the phases sequentially:

Phase 1: MCP Analysis
- Use Phase1_1_InstallMCPTool to install and test the user's MCP locally
- Use Phase1_2_ListMCPToolsTool to discover all available tools and schemas

Phase 2: Workflow Design
- Use Phase2_1_CallWorkflowDesignerSubagentTool to analyze tools and design workflow types
- Use Phase2_2_CallAgentDesignerSubagentTool to create agent configs for each workflow

Return the project directory path when complete."""
    
    def _parse_result(self, result: str, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse MCPAgent result into structured response.
        
        Args:
            result: Raw result from MCPAgent execution
            mcp_config: Original MCP config for context
            
        Returns:
            Structured result dictionary
        """
        logger.debug("Parsing MCPAgent result")
        
        try:
            if self._is_successful_result(result):
                return self._build_success_response(result, mcp_config)
            else:
                return self._build_partial_response(result, mcp_config)
                
        except Exception as e:
            return self._build_error_response(e, result, mcp_config)
    
    def _build_error_response(self, error: Exception, result: str, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build error response with proper logging and traceback"""
        logger.error(f"Error parsing result: {str(error)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "result": result,
            "error": f"Failed to parse result: {str(error)}",
            "traceback": traceback.format_exc(),
            "mcp_config": mcp_config
        }
    
    def _is_successful_result(self, result: str) -> bool:
        """Check if result indicates successful package generation"""
        return "package_path" in result and "installation_command" in result
    
    def _build_success_response(self, result: str, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build response for successful workflow generation"""
        logger.info("Workflow package generation successful")
        return {
            "status": "success",
            "result": result,
            "mcp_config": mcp_config,
            "package_generated": True
        }
    
    def _build_partial_response(self, result: str, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build response for partial workflow generation"""
        logger.warning("Workflow generation completed but package path unclear")
        return {
            "status": "partial",
            "result": result,
            "mcp_config": mcp_config,
            "message": "Workflow generation completed but package path unclear"
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        logger.debug("Entering MCPSquaredAgent context")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        logger.debug("Exiting MCPSquaredAgent context")
        if self.client:
            await self.client.close_all_sessions()


# Configuration for MCPSquaredAgent following design document
MCPSQUARED_AGENT_CONFIG = {
    "name": "mcpsquared_workflow_generator",
    "system_prompt": """You are an MCP workflow package generator. You MUST call tools sequentially to generate installable packages.

PHASE 1: MCP Analysis
1. CALL phase1_1_install_mcp_tool with the provided MCP config
2. CALL phase1_2_list_mcp_tools_tool with the MCP config to get tools file path

PHASE 2: Workflow Design  
3. CALL phase2_1_call_workflow_designer_subagent_tool with tools file path to get designs file path
4. CALL phase2_2_call_agent_designer_subagent_tool with designs file path to get configs directory

CRITICAL: Actually CALL each tool and use the returned file paths for subsequent tools. Do not just describe - EXECUTE the tools.

Return ONLY the project directory path when complete - nothing else.""",
    "provider": "openai", 
    "model": "gpt-5-mini",
    "max_steps": 20,
    "tools": [
        # Phase 1: MCP Analysis Tools
        "Phase1_1_InstallMCPTool",           # Install user's MCP locally and test connection
        "Phase1_2_ListMCPToolsTool",         # Discover all available tools and schemas
        
        # Phase 2: Workflow Design Tools  
        "Phase2_1_CallWorkflowDesignerSubagentTool",  # Subagent analyzes tools → designs workflow types
        "Phase2_2_CallAgentDesignerSubagentTool",     # Subagent creates agent configs for each workflow
    ]
}