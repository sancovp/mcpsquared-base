from setuptools import setup, find_packages

setup(
    name="mcpsquared-base",
    version="0.1.0",
    description="Base package for MCPSquared - provides universal loaders/runners and chain selection MCP server",
    packages=find_packages(),
    install_requires=[
        "mcp-use",
        "langchain-openai",
        "fastmcp",
    ],
    entry_points={
        "console_scripts": [
            "chain-selection-mcp-server=mcpsquared_base.chain_selection_mcp_server:main",
        ],
    },
    python_requires=">=3.8",
)