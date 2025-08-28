"""
Agent Loader - Universal JSON config loading system for MCP agents

This module provides the foundational functionality to load and instantiate
MCPAgents from JSON configuration files.
"""

import json
import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path
from mcp_use import MCPClient, MCPAgent

logger = logging.getLogger(__name__)


def load_agent_from_json_config(config_path: str, input_prompt: str = None) -> MCPAgent:
    """Load and instantiate an MCPAgent from JSON configuration"""
    logger.info(f"Loading agent from config: {config_path}")
    config = _load_json_config(config_path)
    return _create_agent_from_config(config)


def _get_mcp_client_from_config(config: Dict[str, Any]) -> Optional[MCPClient]:
    """Extract and create MCP client based on config format"""
    if config.get("client"):
        # Client config from workflow merge
        return _create_mcp_client(config["client"].get("mcpServers", {}))
    elif config.get("mcp_names"):
        # New format: resolve MCP names from registry
        mcp_servers = _resolve_mcp_names_to_configs(config["mcp_names"])
        return _create_mcp_client(mcp_servers)
    elif config.get("mcp_servers"):
        # Legacy format: direct mcp_servers in config
        return _create_mcp_client(config["mcp_servers"])
    else:
        return _create_mcp_client({})


def _build_mcp_agent(llm, client, config: Dict[str, Any]) -> MCPAgent:
    """Build MCPAgent with provided components"""
    return MCPAgent(
        llm=llm,
        client=client,
        use_server_manager=False,
        system_prompt=config["system_prompt"],
        max_steps=config.get("max_steps", 20)
    )


def _create_agent_from_config(config: Dict[str, Any]) -> MCPAgent:
    """Create MCPAgent instance from validated config"""
    _validate_agent_config(config)
    
    client = _get_mcp_client_from_config(config)
    llm = _create_llm(config)
    agent = _build_mcp_agent(llm, client, config)
    
    logger.info(f"Successfully loaded agent: {config.get('agent_name', 'unnamed')}")
    return agent


def _load_json_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration from file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.debug(f"Loaded config from {config_path}")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")


def _validate_agent_config(config: Dict[str, Any]) -> None:
    """Validate that agent config has required fields"""
    required_fields = ["system_prompt", "model"]
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in agent config: {missing_fields}")


def _create_mcp_client(mcp_servers: Dict[str, Any]) -> Optional[MCPClient]:
    """Create MCP client from server configurations"""
    if not mcp_servers:
        logger.debug("No MCP servers specified, creating agent without MCP client")
        return None
    
    logger.debug(f"Creating MCP client with {len(mcp_servers)} servers")
    client_config = {"mcpServers": mcp_servers}
    return MCPClient.from_dict(client_config)


def _resolve_mcp_names_to_configs(mcp_names: list) -> Dict[str, Any]:
    """Resolve MCP names to actual server configs from registry"""
    logger.debug(f"Resolving MCP names to configs: {mcp_names}")
    
    mcp_servers = {}
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        raise ValueError("WORK_DIR environment variable not set")
    registry_dir = Path(f"{work_dir}/mcp_configs")
    
    for mcp_name in mcp_names:
        config_file = registry_dir / f"{mcp_name}_config.json"
        
        if not config_file.exists():
            logger.warning(f"MCP config not found in registry: {config_file}")
            continue
            
        try:
            with open(config_file, 'r') as f:
                mcp_config = json.load(f)
            mcp_servers[mcp_name] = mcp_config
            logger.debug(f"Loaded MCP config for: {mcp_name}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to load MCP config {config_file}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    logger.info(f"Resolved {len(mcp_servers)} MCP configs from registry")
    return mcp_servers


def _get_provider_key_mapping() -> Dict[str, str]:
    """Get mapping of providers to environment variable names"""
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY"
    }


def _get_api_key_for_provider(provider: str) -> Optional[str]:
    """Get API key from environment for the given provider"""
    import os
    provider_key_map = _get_provider_key_mapping()
    env_var_name = provider_key_map.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(env_var_name)
    
    if api_key:
        logger.debug(f"Using {env_var_name} from environment")
    else:
        logger.warning(f"No {env_var_name} found in environment")
    
    return api_key


def _create_llm(config: Dict[str, Any]):
    """Create LLM instance from config - matches MCPSquaredAgent pattern (no explicit API key)"""
    model = config["model"]
    provider = config.get("provider", "openai")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        logger.debug(f"Creating OpenAI LLM with model: {model} - relying on environment for API key")
        return ChatOpenAI(model=model)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = _get_api_key_for_provider(provider)
        if api_key:
            return ChatAnthropic(model=model, api_key=api_key)
        else:
            logger.warning(f"No API key found for Anthropic")
            return ChatAnthropic(model=model)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = _get_api_key_for_provider(provider)
        if api_key:
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        else:
            logger.warning(f"No API key found for Google")
            return ChatGoogleGenerativeAI(model=model)
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        api_key = _get_api_key_for_provider(provider)
        if api_key:
            return AzureChatOpenAI(model=model, api_key=api_key)
        else:
            logger.warning(f"No API key found for Azure")
            return AzureChatOpenAI(model=model)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def load_agent_from_dict_config(config: Dict[str, Any]) -> MCPAgent:
    """Load agent from dictionary config (for in-memory configs)"""
    logger.info(f"Loading agent from dict config: {config.get('name', 'unnamed')}")
    return _create_agent_from_config(config)