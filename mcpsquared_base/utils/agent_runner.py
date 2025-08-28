"""
Agent Runner - Low-level agent execution

Provides pure agent execution functionality: takes a complete config and prompt,
runs the agent, returns raw result without any postprocessing.
"""

import logging
from typing import Dict, Any
from mcp_use import MCPAgent
from .agent_loader import load_agent_from_dict_config

logger = logging.getLogger(__name__)


async def agent_runner(config: Dict[str, Any], prompt: str) -> str:
    """
    Execute an agent with complete configuration and prompt
    
    Args:
        config: Complete agent configuration (no blanks)
        prompt: Input prompt to run the agent with
        
    Returns:
        Raw result from agent execution (no postprocessing)
        
    Raises:
        Exception: If agent execution fails
    """
    logger.info(f"Running agent: {config.get('name', 'unnamed')}")
    logger.debug(f"Agent config: {config}")
    logger.debug(f"Input prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Input prompt: {prompt}")
    
    try:
        # Load agent from complete config
        agent = load_agent_from_dict_config(config)
        
        # Run agent with prompt
        logger.debug("Executing agent...")
        result = await agent.run(prompt)
        
        logger.info("Agent execution completed successfully")
        logger.debug(f"Raw result: {result[:200]}..." if len(str(result)) > 200 else f"Raw result: {result}")
        
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"Agent execution failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Cleanup MCP client sessions if agent was created
        if 'agent' in locals() and hasattr(agent, 'client') and agent.client:
            try:
                await agent.client.close_all_sessions()
                logger.debug("Cleaned up MCP client sessions")
            except Exception as cleanup_error:
                import traceback
                logger.warning(f"Failed to cleanup MCP sessions: {cleanup_error}")
                logger.warning(f"Cleanup traceback: {traceback.format_exc()}")


def _validate_complete_config(config: Dict[str, Any]) -> None:
    """Validate that config has no blank fields"""
    blank_fields = []
    
    for key, value in config.items():
        if value is None and key in ['client', 'disallowed_tools']:
            blank_fields.append(key)
    
    if blank_fields:
        raise ValueError(f"Config has blank fields that should be filled by workflow: {blank_fields}")
    
    # Also validate required fields exist
    required_fields = ["system_prompt", "model"]
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in config: {missing_fields}")