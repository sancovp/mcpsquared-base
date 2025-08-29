#!/usr/bin/env python3
"""
Chain Selection Tool - Writes chain selections to handoff files

This tool allows selector agents to write their chain selections to temporary files
for handoff to executor agents. Uses agent name + timestamp naming for parallel execution.
"""

import json
import os
import time
import glob
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _ensure_selection_directory() -> str:
    """Ensure chain selection directory exists and return path."""
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        raise ValueError("WORK_DIR environment variable not set")
    selection_dir = f"{work_dir}/chain_selections"
    os.makedirs(selection_dir, exist_ok=True)
    return selection_dir


def _generate_selection_filepath(agent_name: str, selection_dir: str) -> str:
    """Generate unique filepath for chain selection."""
    timestamp = int(time.time() * 1000)
    filename = f"{agent_name}_{timestamp}.json"
    return os.path.join(selection_dir, filename)


def _build_selection_data(agent_name: str, chain_sequence: List[str], reasoning: str) -> Dict[str, Any]:
    """Build selection data dictionary."""
    return {
        "agent_name": agent_name,
        "timestamp": int(time.time() * 1000),
        "chain_sequence": chain_sequence,
        "reasoning": reasoning,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }


def write_chain_selection(agent_name: str, chain_sequence: List[str], reasoning: str) -> str:
    """
    Write chain selection to handoff file with agent name and timestamp.
    
    Args:
        agent_name: Name of the agent making the selection (for parallel execution)
        chain_sequence: List of phase tools to execute in order
        reasoning: Explanation of why this sequence was chosen
        
    Returns:
        Path to the written selection file
    """
    selection_dir = _ensure_selection_directory()
    filepath = _generate_selection_filepath(agent_name, selection_dir)
    selection_data = _build_selection_data(agent_name, chain_sequence, reasoning)
    
    with open(filepath, 'w') as f:
        json.dump(selection_data, f, indent=2)
    
    logger.debug(f"Chain selection written to {filepath}")
    return filepath


def _find_chain_files(agent_name: str) -> List[str]:
    """Find all chain selection files for an agent."""
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        raise ValueError("WORK_DIR environment variable not set")
    selection_dir = f"{work_dir}/chain_selections"
    pattern = os.path.join(selection_dir, f"{agent_name}_*.json")
    return glob.glob(pattern)


def _get_most_recent_file(files: List[str]) -> str:
    """Get the most recent file by timestamp in filename."""
    return max(files, key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))


def get_last_chain_file(agent_name: str) -> str:
    """
    Get the most recent chain selection file for a specific agent.
    
    Args:
        agent_name: Name of the agent to get selection for
        
    Returns:
        Path to the most recent chain selection file
        
    Raises:
        FileNotFoundError: If no chain files exist for the agent
    """
    files = _find_chain_files(agent_name)
    
    if not files:
        raise FileNotFoundError(f"No chain selection files found for agent '{agent_name}'")
    
    return _get_most_recent_file(files)


def _extract_timestamp_from_filename(filename: str) -> Optional[int]:
    """Extract timestamp from chain selection filename"""
    import re
    match = re.search(r'_(\d+)\.json$', filename)
    return int(match.group(1)) if match else None


def _find_file_with_highest_timestamp(files: list) -> str:
    """Find the file with the highest timestamp in its filename"""
    latest_timestamp = 0
    latest_file = None
    
    for file_path in files:
        timestamp = _extract_timestamp_from_filename(file_path.name)
        if timestamp and timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_file = file_path
    
    if latest_file is None:
        raise FileNotFoundError("No valid chain selection files found with timestamps")
    
    return str(latest_file)


def get_most_recent_chain_file() -> str:
    """
    Get the most recent chain selection file regardless of agent name.
    
    Returns:
        Path to the most recent chain selection file
        
    Raises:
        FileNotFoundError: If no chain files exist at all
    """
    from pathlib import Path
    
    selection_dir = Path(_ensure_selection_directory())
    all_files = list(selection_dir.glob("*.json"))
    
    if not all_files:
        raise FileNotFoundError("No chain selection files found")
    
    return _find_file_with_highest_timestamp(all_files)


def read_chain_selection(filepath: str) -> Dict[str, Any]:
    """
    Read and parse a chain selection file.
    
    Args:
        filepath: Path to the chain selection file
        
    Returns:
        Parsed chain selection data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_chain_for_executor(request: Dict[str, Any], chain_file: str) -> str:
    """
    Format the selected chain into markdown for executor prompt.
    
    Args:
        request: Original request data
        chain_file: Path to chain selection file
        
    Returns:
        Formatted prompt with chain selection for executor
    """
    selection_data = read_chain_selection(chain_file)
    
    chain_markdown = f"""
## Selected Chain Sequence

**Reasoning:** {selection_data['reasoning']}

**Chain to Execute:**
"""
    
    for i, phase in enumerate(selection_data['chain_sequence'], 1):
        chain_markdown += f"{i}. {phase}\n"
    
    chain_markdown += f"""
**Original Request:** {json.dumps(request, indent=2)}

Execute this chain sequence step by step, using the results from each phase as input to the next.
"""
    
    return chain_markdown


# Cleanup utility for long-running systems
def cleanup_old_selections(max_age_hours: int = 24):
    """
    Clean up old chain selection files to prevent disk usage growth.
    
    Args:
        max_age_hours: Maximum age of files to keep (default 24 hours)
    """
    work_dir = os.getenv("WORK_DIR")
    if not work_dir:
        raise ValueError("WORK_DIR environment variable not set")
    selection_dir = f"{work_dir}/chain_selections"
    if not os.path.exists(selection_dir):
        return
    
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    for filename in os.listdir(selection_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(selection_dir, filename)
        try:
            # Extract timestamp from filename
            timestamp = int(filename.split('_')[-1].split('.')[0]) / 1000
            if timestamp < cutoff_time:
                os.remove(filepath)
                print(f"Cleaned up old selection file: {filename}")
        except (ValueError, IndexError):
            # Skip files that don't match our naming convention
            continue