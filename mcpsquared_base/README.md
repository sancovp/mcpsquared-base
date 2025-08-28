# Chain Selection MCP Server

Provides tools for MCPSquared selector/executor handoff coordination.

## Features

- **write_chain_selection_tool**: Write chain selections to handoff files
- Agent-specific file naming with timestamps for parallel execution
- Automatic cleanup utilities for long-running systems

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run as MCP server:

```bash
python server.py
```

Or use the entry point:

```bash
chain-selection-mcp-server
```

## Configuration

Add to your MCP client config:

```json
{
  "command": "python",
  "args": ["server.py"],
  "transport": "stdio"
}
```

## File Storage

Chain selections are stored in `/tmp/mcpsquared/chain_selections/` with format:
`{agent_name}_{timestamp}.json`