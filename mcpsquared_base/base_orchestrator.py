"""
MCPSquaredOrchestrator - Main orchestration logic for MCPSquared workflow generation

This orchestrator manages the lifecycle of MCPSquaredAgent instances and provides
the main entry point for the FastMCP server tools.
"""

from typing import Dict, Any, Optional
import logging
import traceback
import json
from .mcpsquared_agent import MCPSquaredAgent, MCPSQUARED_AGENT_CONFIG
from .registry.registry_tools import (
    get_processed_mcps_list as get_mcps_list,
    search_my_workflows as search_workflows_tool,
    execute_any_workflow as execute_workflow_tool
)

# Setup logging
logger = logging.getLogger(__name__)


class MCPSquaredOrchestrator:
    """
    Main orchestrator for MCPSquared workflow generation.
    
    Manages MCPSquaredAgent lifecycle and provides unified interface for FastMCP tools.
    """
    
    def __init__(self):
        """Initialize orchestrator with default configuration"""
        self.agent_config = self._load_agent_config()
        logger.info("MCPSquaredOrchestrator initialized")
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """
        Load MCPSquaredAgent configuration.
        
        Returns:
            Agent configuration dictionary
        """
        logger.debug("Loading MCPSquaredAgent configuration")
        return MCPSQUARED_AGENT_CONFIG.copy()
    
    async def run_workflow(self, workflow_config: Dict[str, Any]) -> str:
        """
        Main entry point for workflow generation.
        
        Args:
            workflow_config: MCP configuration to generate workflows from
            
        Returns:
            JSON string with generation results
        """
        logger.info(f"Starting workflow generation for: {workflow_config.get('name', 'unknown')}")
        
        try:
            # Create and execute MCPSquaredAgent
            result = await self._execute_workflow_generation(workflow_config)
            return self._format_success_response(result)
            
        except Exception as e:
            logger.error(f"Workflow generation failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._format_error_response(e, workflow_config)
    
    async def _execute_workflow_generation(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow generation using MCPSquaredAgent.
        
        Args:
            workflow_config: MCP configuration
            
        Returns:
            Generation result dictionary
        """
        logger.debug("Creating MCPSquaredAgent for workflow generation")
        
        # Use async context manager for proper cleanup
        async with MCPSquaredAgent(self.agent_config) as agent:
            result = await agent.generate_workflow_package(workflow_config)
            logger.debug("Workflow generation completed")
            return result
    
    def _format_success_response(self, result: Dict[str, Any]) -> str:
        """
        Format successful workflow generation response.
        
        Args:
            result: Generation result from MCPSquaredAgent
            
        Returns:
            JSON formatted success response
        """
        logger.info("Formatting successful workflow generation response")
        
        response = self._build_base_success_response(result)
        
        # Add package-specific info if available
        if result.get("package_generated"):
            self._add_package_info(response, result)
        
        return json.dumps(response, indent=2)
    
    def _build_base_success_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build base success response structure"""
        return {
            "success": True,
            "status": result.get("status", "unknown"),
            "package_generated": result.get("package_generated", False),
            "result": result.get("result", ""),
            "mcp_config": result.get("mcp_config", {})
        }
    
    def _extract_package_details(self, result_text: str) -> tuple:
        """Extract package path and name from result text"""
        import re
        path_match = re.search(r"Package path: ([^\n]+)", result_text)
        name_match = re.search(r"Package name: ([^\n]+)", result_text)
        
        package_path = path_match.group(1).strip() if path_match else "Unknown"
        package_name = name_match.group(1).strip() if name_match else "Unknown"
        
        return package_path, package_name

    def _add_package_info(self, response: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Add package-specific information to response"""
        logger.debug("Adding package information to response")
        
        result_text = result.get("result", "")
        
        # Extract package information from result text - look for actual patterns
        if "Package path:" in result_text and "Package name:" in result_text:
            package_path, package_name = self._extract_package_details(result_text)
            
            response["package_info"] = {
                "generation_completed": True,
                "package_path": package_path,
                "package_name": package_name,
                "installation_command": f"pip install {package_path}"
            }
            logger.info(f"Package generation successful: {package_name} at {package_path}")
        else:
            logger.warning("Package information not found in result text")
    
    def _format_error_response(self, error: Exception, workflow_config: Dict[str, Any]) -> str:
        """
        Format error response for failed workflow generation.
        
        Args:
            error: Exception that occurred
            workflow_config: Original workflow configuration
            
        Returns:
            JSON formatted error response
        """
        logger.error("Formatting error response for workflow generation failure")
        
        response = self._build_base_error_response(error, workflow_config)
        return json.dumps(response, indent=2)
    
    def _build_base_error_response(self, error: Exception, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build base error response structure"""
        return {
            "success": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "mcp_config": workflow_config,
            "suggestions": [
                "Check MCP configuration format",
                "Ensure MCP server is accessible",
                "Verify phase tools are properly installed"
            ]
        }
    
    def get_flows(self, domain: Optional[str] = None) -> str:
        """
        List available generated workflows.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            JSON string with available workflows
        """
        logger.info(f"Listing workflows for domain: {domain or 'all'}")
        return search_workflows_tool(domain=domain)
    
    def get_flow_domains(self) -> str:
        """
        List all available workflow domains.
        
        Returns:
            JSON string with available domains
        """
        logger.info("Listing workflow domains")
        
        # TODO: Implement domain scanning logic
        response = {
            "success": True,
            "domains": [],
            "message": "Domain listing not yet implemented - this is v1.0 MVP"
        }
        
        return json.dumps(response, indent=2)
    
    def sub_chat(self, message: str, history_id: str) -> str:
        """
        Chat interface for workflow planning.
        
        Args:
            message: User message
            history_id: Conversation ID
            
        Returns:
            JSON string with chat response
        """
        logger.info(f"Processing chat message for session: {history_id}")
        
        response = self._build_chat_response(history_id)
        return json.dumps(response, indent=2)
    
    def _build_chat_response(self, history_id: str) -> Dict[str, Any]:
        """Build chat response structure"""
        return {
            "success": True,
            "response": "MCPSquared v1.0 is focused on workflow generation. Use generate_flows_for_mcp() to create workflow packages.",
            "history_id": history_id,
            "recommendations": [
                "Provide your MCP configuration to generate_flows_for_mcp()",
                "Ensure your MCP server is accessible before generation",
                "Generated packages will be PyPI-installable"
            ]
        }
    
    def package_workflow_mcp(self, workflows: list, base_mcp: str) -> str:
        """
        Package specific workflows into installable MCP.
        
        Args:
            workflows: List of workflow names
            base_mcp: Base MCP name
            
        Returns:
            JSON string with packaging results
        """
        logger.info(f"Packaging workflows {workflows} for base MCP: {base_mcp}")
        
        # TODO: Implement custom packaging logic
        response = {
            "success": True,
            "workflows": workflows,
            "base_mcp": base_mcp,
            "message": "Custom workflow packaging not yet implemented - this is v1.0 MVP"
        }
        
        return json.dumps(response, indent=2)
    
    def get_processed_mcps_list(self) -> str:
        """
        Get list of all MCPs that have been processed by MCPSquared
        
        Returns:
            JSON string with list of MCP names and basic stats
        """
        logger.info("Getting processed MCPs list")
        return get_mcps_list()
    
    def search_my_workflows(
        self, 
        mcp_name: Optional[str] = None,
        domain: Optional[str] = None, 
        workflow_name_pattern: Optional[str] = None
    ) -> str:
        """
        Search workflows across all processed MCPs
        
        Args:
            mcp_name: Filter by specific MCP name
            domain: Filter by workflow domain
            workflow_name_pattern: Search for workflows containing this text
            
        Returns:
            JSON string with search results
        """
        logger.info(f"Searching workflows with filters: mcp={mcp_name}, domain={domain}, pattern={workflow_name_pattern}")
        return search_workflows_tool(mcp_name, domain, workflow_name_pattern)
    
    def execute_any_workflow(
        self,
        workflow_name: str,
        workflow_args: Dict[str, Any],
        package_name: Optional[str] = None
    ) -> str:
        """
        Execute any workflow from the registry by name
        
        Args:
            workflow_name: Name of the workflow to execute
            workflow_args: Arguments to pass to the workflow
            package_name: Optional package name to narrow search
            
        Returns:
            JSON string with execution results
        """
        logger.info(f"Executing workflow: {workflow_name} from package: {package_name or 'any'}")
        return execute_workflow_tool(workflow_name, workflow_args, package_name)
    
    async def domain_specific_agent(self, prompt: str) -> str:
        """
        Domain-specific wrapper for MCPSquared workflow generation.
        
        This method provides pre/post hooks around the core MCPSquaredAgent.run() call.
        When morphing MCPSquared for other domains, edit this method to customize
        domain-specific setup, teardown, and agent orchestration logic.
        
        Args:
            prompt: User prompt/request for workflow generation
            
        Returns:
            JSON string with domain-specific workflow generation results
        """
        logger.info(f"Starting domain-specific agent workflow for prompt: {prompt[:100]}...")
        
        try:
            # PRE-HOOKS: Domain-specific setup before agent execution
            pre_hook_result = await self._execute_pre_hooks(prompt)
            
            # CORE AGENT CALL: Universal MCPSquaredAgent execution
            agent_request = self._build_agent_request(prompt, pre_hook_result)
            agent_result = await self._execute_core_agent(agent_request)
            
            # POST-HOOKS: Domain-specific processing after agent execution
            final_result = await self._execute_post_hooks(agent_result, prompt)
            
            return self._format_domain_specific_response(final_result)
            
        except Exception as e:
            logger.error(f"Domain-specific agent workflow failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._format_domain_specific_error_response(e, prompt)
    
    async def _execute_pre_hooks(self, prompt: str) -> Dict[str, Any]:
        """
        Execute domain-specific pre-hooks before agent execution.
        
        FOR MCPSQUARED DOMAIN: Install MCP and discover tools without LLM overhead.
        FOR OTHER DOMAINS: Customize this method with domain-specific setup logic.
        
        Args:
            prompt: User prompt to analyze and extract MCP config from
            
        Returns:
            Pre-hook results to pass to core agent
        """
        logger.info("Executing pre-hooks for MCPSquared domain")
        
        # TODO: Extract MCP config from prompt
        extracted_mcp_config = self._extract_mcp_config_from_prompt(prompt)
        
        # TODO: Implement Phase 1.1 - Install MCP locally and test connection
        phase_1_1_result = await self._install_mcp_locally(extracted_mcp_config)
        
        # TODO: Implement Phase 1.2 - Discover all available tools and schemas  
        phase_1_2_result = await self._discover_mcp_tools(extracted_mcp_config, phase_1_1_result)
        
        return {
            "extracted_mcp_config": extracted_mcp_config,
            "phase_1_1": phase_1_1_result,
            "phase_1_2": phase_1_2_result,
            "available_tools": phase_1_2_result.get("tools", []),
            "tool_schemas": phase_1_2_result.get("schemas", {})
        }
    
    async def _execute_core_agent(self, agent_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the universal MCPSquaredAgent.run() method.
        
        This stays consistent across all morphed versions - only the wrapper changes.
        
        Args:
            agent_request: Request built from prompt + pre_hook_result
            
        Returns:
            Result from MCPSquaredAgent.run()
        """
        logger.info("Executing core MCPSquaredAgent.run() method")
        
        # Create and execute MCPSquaredAgent with new run() method
        async with MCPSquaredAgent(self.agent_config) as agent:
            result = await agent.run(agent_request)
            logger.debug("Core agent execution completed")
            return result
    
    async def _execute_post_hooks(self, agent_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Execute domain-specific post-hooks after agent execution.
        
        FOR MCPSQUARED DOMAIN: Package workflows into installable format.
        FOR OTHER DOMAINS: Customize this method with domain-specific cleanup logic.
        
        Args:
            agent_result: Result from core agent execution
            prompt: Original user prompt for context
            
        Returns:
            Final processed result
        """
        logger.info("Executing post-hooks for MCPSquared domain")
        
        # TODO: Implement workflow packaging and finalization
        packaging_result = await self._package_workflows(agent_result, prompt)
        
        return {
            "agent_result": agent_result,
            "packaging_result": packaging_result,
            "final_status": "complete" if packaging_result.get("success") else "partial"
        }
    
    def _build_agent_request(self, prompt: str, pre_hook_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build agent request from prompt and pre-hook results.
        
        Args:
            prompt: Original user prompt
            pre_hook_result: Results from pre-hooks execution
            
        Returns:
            Formatted request for MCPSquaredAgent.run()
        """
        logger.debug("Building agent request from prompt and pre-hook results")
        
        return {
            "user_prompt": prompt,
            "mcp_config": pre_hook_result.get("extracted_mcp_config", {}),
            "available_tools": pre_hook_result.get("available_tools", []),
            "tool_schemas": pre_hook_result.get("tool_schemas", {}),
            "pre_analysis": {
                "installation_status": pre_hook_result.get("phase_1_1", {}),
                "tool_discovery": pre_hook_result.get("phase_1_2", {})
            }
        }
    
    def _format_domain_specific_response(self, final_result: Dict[str, Any]) -> str:
        """
        Format the final domain-specific response.
        
        Args:
            final_result: Final result from post-hooks
            
        Returns:
            JSON formatted response
        """
        logger.info("Formatting domain-specific response")
        
        response = {
            "success": final_result.get("final_status") == "complete",
            "status": final_result.get("final_status", "unknown"),
            "agent_result": final_result.get("agent_result", {}),
            "packaging_info": final_result.get("packaging_result", {}),
            "domain": "mcpsquared_workflow_generation"
        }
        
        return json.dumps(response, indent=2)
    
    def _format_domain_specific_error_response(self, error: Exception, prompt: str) -> str:
        """
        Format error response for domain-specific agent failures.
        
        Args:
            error: Exception that occurred
            prompt: Original user prompt
            
        Returns:
            JSON formatted error response
        """
        response = self._build_error_response_structure(error, prompt)
        return json.dumps(response, indent=2)
    
    def _build_error_response_structure(self, error: Exception, prompt: str) -> Dict[str, Any]:
        """Build error response structure for domain-specific failures"""
        return {
            "success": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "prompt": prompt,
            "domain": "mcpsquared_workflow_generation",
            "suggestions": self._get_error_suggestions()
        }
    
    def _get_error_suggestions(self) -> list:
        """Get standard error suggestions for domain-specific failures"""
        return [
            "Check prompt format for MCP configuration",
            "Ensure MCP server is accessible",
            "Verify phase tools are properly installed"
        ]
    
    # Helper method stubs
    def _extract_mcp_config_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        STUB: Extract MCP configuration from user prompt.
        
        TODO: Implement prompt parsing to extract MCP config details.
        """
        logger.info("STUB: Extracting MCP config from prompt")
        return {
            "name": "extracted_mcp_name",
            "command": "python",
            "args": ["-m", "extracted_mcp_module"],
            "transport": "stdio"
        }
    
    async def _install_mcp_locally(self, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUB: Install MCP locally and test connection (Phase 1.1 equivalent).
        
        TODO: Implement deterministic MCP installation and connection testing.
        """
        logger.info(f"STUB: Installing MCP locally: {mcp_config.get('name', 'unknown')}")
        return {
            "success": True,
            "message": "STUB: MCP installation successful",
            "mcp_name": mcp_config.get("name"),
            "connection_tested": True
        }
    
    async def _discover_mcp_tools(self, mcp_config: Dict[str, Any], install_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        STUB: Discover all available tools and schemas (Phase 1.2 equivalent).
        
        TODO: Implement deterministic tool discovery without LLM overhead.
        """
        logger.info(f"STUB: Discovering tools for MCP: {mcp_config.get('name', 'unknown')}")
        return {
            "success": True,
            "message": "STUB: Tool discovery successful", 
            "tools": ["stub_tool_1", "stub_tool_2", "stub_tool_3"],
            "schemas": {
                "stub_tool_1": {"type": "function", "description": "Stub tool 1"},
                "stub_tool_2": {"type": "function", "description": "Stub tool 2"},
                "stub_tool_3": {"type": "function", "description": "Stub tool 3"}
            }
        }
    
    async def _package_workflows(self, agent_result: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        STUB: Package workflows into installable format.
        
        TODO: Implement workflow packaging and PyPI preparation.
        """
        logger.info("STUB: Packaging workflows into installable format")
        return {
            "success": True,
            "message": "STUB: Workflow packaging successful",
            "package_path": "/tmp/stub_package_path",
            "package_name": f"workflows_for_extracted_mcp",
            "installation_command": "pip install /tmp/stub_package_path"
        }