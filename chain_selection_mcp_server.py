#!/usr/bin/env python3
"""
Chain Selection MCP Server

Provides tools for MCPSquaredAgent selector/executor handoff:
- write_chain_selection: Write chain selections to handoff files
"""

import logging
from typing import List
from fastmcp import FastMCP
from mcpsquared_base.tools.chain_selection import write_chain_selection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("Chain Selection MCP")

def _log_chain_selection_request(agent_name: str, chain_sequence: List[str], reasoning: str):
    """Log incoming chain selection request details."""
    logger.info(f"Writing chain selection for agent '{agent_name}' with {len(chain_sequence)} phases")
    logger.debug(f"Chain sequence: {chain_sequence}")
    logger.debug(f"Reasoning: {reasoning}")


def _handle_chain_selection_write(agent_name: str, chain_sequence: List[str], reasoning: str) -> str:
    """Handle the actual chain selection writing with error handling."""
    try:
        filepath = write_chain_selection(agent_name, chain_sequence, reasoning)
        logger.info(f"Chain selection written successfully to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to write chain selection: {str(e)}")
        raise


@mcp.tool()
def write_chain_selection_tool(agent_name: str, chain_sequence: List[str], reasoning: str) -> str:
    """
    Write chain selection to handoff file for selector/executor coordination.
    
    Args:
        agent_name: Name of the agent making the selection (enables parallel execution)
        chain_sequence: List of phase tools to execute in order (e.g., ["phase1_1", "phase1_2"])
        reasoning: Explanation of why this sequence was chosen
        
    Returns:
        Path to the written chain selection file
    """
    _log_chain_selection_request(agent_name, chain_sequence, reasoning)
    return _handle_chain_selection_write(agent_name, chain_sequence, reasoning)


def main():
    """Main entry point for console script"""
    logger.info("Starting Chain Selection MCP Server")
    mcp.run()

if __name__ == "__main__":
    main()