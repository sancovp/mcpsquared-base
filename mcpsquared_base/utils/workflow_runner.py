"""
Workflow Runner - High-level workflow execution

Orchestrates workflow execution by loading configs, merging them,
formatting prompts, and calling the agent runner.
"""

import json
import logging
import traceback
import os
from typing import Dict, Any
from pathlib import Path
from .agent_runner import agent_runner

logger = logging.getLogger(__name__)


async def run_workflow(workflow_name: str, workflow_args: Dict[str, Any], allowed_tools: list = None, config_dir: str = None) -> str:
    """
    Execute a workflow by name with provided arguments
    
    Args:
        workflow_name: Name of workflow to execute
        workflow_args: Arguments to pass to workflow
        allowed_tools: Optional list of tools to limit agent to (acts as sequence/filter)
        config_dir: Optional path to directory containing workflow config files (e.g., "/path/to/configs/workflows")
        
    Returns:
        Formatted result from workflow execution
        
    Raises:
        Exception: If workflow execution fails
    """
    logger.info(f"Running workflow: {workflow_name}")
    logger.debug(f"Workflow args: {workflow_args}")
    if allowed_tools:
        logger.debug(f"Allowed tools override: {allowed_tools}")
    
    try:
        # Load workflow config
        workflow_config = _load_workflow_config(workflow_name, config_dir)
        
        # Load agent config
        agent_config = _load_agent_config(workflow_config["agent_config_name"], config_dir)
        
        # Merge configs
        complete_config = _merge_configs(workflow_config, agent_config)
        
        # Apply allowed_tools override if provided
        if allowed_tools:
            complete_config["allowed_tools"] = allowed_tools
            logger.info(f"Applied allowed_tools override: {len(allowed_tools)} tools")
        
        # Format prompt
        formatted_prompt = _format_prompt(workflow_config, workflow_args)
        
        # Run agent
        raw_result = await agent_runner(complete_config, formatted_prompt)
        
        # Format result
        formatted_result = _format_result(raw_result, workflow_name)
        
        logger.info(f"Workflow {workflow_name} completed successfully")
        return formatted_result
        
    except Exception as e:
        logger.error(f"Workflow {workflow_name} failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def _load_workflow_config(workflow_name: str, config_dir: str = None) -> Dict[str, Any]:
    """Load workflow configuration from file"""
    config_path = _get_workflow_config_path(workflow_name, config_dir)
    return _read_config_file(config_path)

def _get_workflow_config_path(workflow_name: str, config_dir: str = None) -> Path:
    """Get the path to workflow configuration file"""
    if config_dir:
        return Path(config_dir) / "workflows" / f"{workflow_name}.json"
    else:
        env_config_dir = os.getenv("MCPSQUARED_CONFIG_DIR")
        if not env_config_dir:
            raise ValueError("MCPSQUARED_CONFIG_DIR environment variable not set")
        return Path(f"{env_config_dir}/workflows/{workflow_name}.json")

def _read_config_file(config_path: Path) -> Dict[str, Any]:
    """Read and parse JSON configuration file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.debug(f"Loaded config from: {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config {config_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Invalid JSON in config {config_path}: {e}")


def _load_agent_config(agent_config_name: str, config_dir: str = None) -> Dict[str, Any]:
    """Load agent configuration from file"""
    config_path = _get_agent_config_path(agent_config_name, config_dir)
    return _read_config_file(config_path)

def _get_agent_config_path(agent_config_name: str, config_dir: str = None) -> Path:
    """Get the path to agent configuration file"""
    if config_dir:
        return Path(config_dir) / "agents" / f"{agent_config_name}.json"
    else:
        env_config_dir = os.getenv("MCPSQUARED_CONFIG_DIR")
        if not env_config_dir:
            raise ValueError("MCPSQUARED_CONFIG_DIR environment variable not set")
        return Path(f"{env_config_dir}/agents/{agent_config_name}.json")


def _merge_configs(workflow_config: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge workflow config into agent config (workflow fills the blanks)"""
    logger.debug("Merging workflow and agent configs")
    
    # Start with agent config
    complete_config = agent_config.copy()
    
    # Fill in blanks from workflow config
    if workflow_config.get("mcp_client_config"):
        # Expand environment variables in the client config (SAME AS FIRST DRAFT)
        client_config = workflow_config["mcp_client_config"].copy()
        _expand_env_vars_recursively(client_config)
        complete_config["client"] = client_config
    
    if workflow_config.get("disallowed_tools"):
        complete_config["disallowed_tools"] = workflow_config["disallowed_tools"]
    
    if workflow_config.get("additional_instructions"):
        complete_config["additional_instructions"] = workflow_config["additional_instructions"]
    
    logger.debug("Config merge completed")
    return complete_config


def _add_api_keys_to_mcp_servers(client_config: Dict[str, Any]) -> None:
    """Process MCP server configs using EXACT SAME logic as MCPSquaredAgent"""
    if "mcpServers" not in client_config:
        return
    
    # Check if we need to use MCPSquaredAgent's config logic
    for server_name, server_config in client_config["mcpServers"].items():
        if server_config == "USE_MCPSQUAREDAGENT_GET_PHASE_TOOLS_CONFIG":
            # Import and use MCPSquaredAgent's exact config creation
            from core.mcpsquared_agent import MCPSquaredAgent
            
            # Create temporary MCPSquaredAgent instance to use its config method
            temp_config = {"provider": "openai", "model": "gpt-5-mini"}
            temp_agent = MCPSquaredAgent.__new__(MCPSquaredAgent)
            temp_agent.config = temp_config
            
            # Get the exact same config MCPSquaredAgent creates
            phase_tools_config = temp_agent._get_phase_tools_config()
            client_config["mcpServers"][server_name] = phase_tools_config
            logger.debug(f"Used MCPSquaredAgent._get_phase_tools_config() for {server_name}")
            break


def _expand_env_vars_recursively(config: Dict[str, Any]) -> None:
    """Recursively expand environment variables in config dict"""
    import re
    
    def expand_value(value):
        if isinstance(value, str):
            # Replace ${VAR_NAME} with environment variable value
            def replace_env_var(match):
                env_var = match.group(1)
                return os.getenv(env_var, match.group(0))  # Keep original if env var not found
            
            return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
        elif isinstance(value, dict):
            for k, v in value.items():
                value[k] = expand_value(v)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                value[i] = expand_value(item)
        return value
    
    expand_value(config)


def _format_prompt(workflow_config: Dict[str, Any], workflow_args: Dict[str, Any]) -> str:
    """Format workflow prompt template with provided arguments"""
    logger.debug("Formatting workflow prompt")
    
    input_prompt = workflow_config["input_prompt"]
    templated_args = workflow_config.get("templated_args", [])
    
    # Replace template variables
    formatted_prompt = input_prompt
    for arg_name in templated_args:
        if arg_name in workflow_args:
            placeholder = f"{{{{{arg_name}}}}}"
            # Use JSON serialization for dicts/lists and escape braces to avoid template parsing
            if isinstance(workflow_args[arg_name], (dict, list)):
                json_str = json.dumps(workflow_args[arg_name])
                # Escape braces to prevent downstream template parsing: { becomes {{, } becomes }}
                replacement = json_str.replace('{', '{{').replace('}', '}}')
            else:
                replacement = str(workflow_args[arg_name])
            formatted_prompt = formatted_prompt.replace(placeholder, replacement)
            logger.debug(f"Replaced {placeholder} with: {replacement[:50]}...")
    
    logger.debug(f"Formatted prompt: {formatted_prompt[:100]}...")
    return formatted_prompt


def _format_result(raw_result: str, workflow_name: str) -> str:
    """Format raw agent result for workflow output"""
    logger.debug(f"Formatting result for workflow: {workflow_name}")
    
    # For now, just return raw result
    # Later we can add workflow-specific result formatting
    return raw_result


def _expand_env_vars(config: Dict[str, Any]) -> None:
    """Recursively expand environment variables in config (in-place)"""
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            # Extract environment variable name
            env_var = value[2:-1]
            config[key] = os.getenv(env_var, value)
            logger.debug(f"Expanded {value} to {config[key][:20]}..." if len(str(config[key])) > 20 else f"Expanded {value} to {config[key]}")
        elif isinstance(value, dict):
            _expand_env_vars(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _expand_env_vars(item)


async def workflow_runner(workflow_name: str, workflow_args: Dict[str, Any]) -> str:
    """
    Alternative entry point - same as run_workflow
    Provided for consistency with naming convention
    """
    return await run_workflow(workflow_name, workflow_args)