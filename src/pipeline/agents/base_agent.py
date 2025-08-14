"""Base agent class for pipeline agents.

This module provides the base functionality that all pipeline agents inherit,
including error handling, logging, and common agent patterns.
"""

import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from smolagents import CodeAgent, HfApiModel
    from transformers import pipeline
except ImportError:
    # Fallback for when dependencies are not available
    CodeAgent = None
    HfApiModel = None


class BasePipelineAgent(ABC):
    """
    Base class for all pipeline agents.
    
    Provides common functionality for error handling, logging,
    and agent orchestration patterns.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", 
                 agent_name: str = "pipeline_agent"):
        """
        Initialize the base pipeline agent.
        
        Args:
            model_id: HuggingFace model ID to use for the agent
            agent_name: Name identifier for this agent
        """
        self.model_id = model_id
        self.agent_name = agent_name
        self.agent = None
        self.execution_history = []
        
        # Initialize the agent if dependencies are available
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the smolagents CodeAgent."""
        try:
            if CodeAgent and HfApiModel:
                model = HfApiModel(model_id=self.model_id)
                self.agent = CodeAgent(
                    tools=self._get_agent_tools(),
                    model=model,
                    max_steps=10
                )
            else:
                self.agent = None
                self._log_warning("smolagents dependencies not available - agent will use fallback mode")
        except Exception as e:
            self.agent = None
            self._log_error(f"Failed to initialize agent: {e}")
    
    @abstractmethod
    def _get_agent_tools(self) -> List:
        """
        Get the list of tools this agent should have access to.
        
        Returns:
            List of tool functions for the agent
        """
        pass
    
    @abstractmethod
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main step for this agent.
        
        Args:
            input_data: Input data for the agent step
            config: Configuration parameters for the step
            
        Returns:
            Dictionary containing the step results
        """
        pass
    
    def run(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with error handling and logging.
        
        Args:
            input_data: Input data for the agent
            config: Configuration for the agent execution
            
        Returns:
            Dictionary containing execution results
        """
        start_time = datetime.now()
        
        try:
            # Log execution start
            self._log_info(f"Starting {self.agent_name} execution")
            
            # Validate inputs
            validation_result = self._validate_inputs(input_data, config)
            if not validation_result["valid"]:
                return self._create_error_result(
                    f"Input validation failed: {validation_result['error']}"
                )
            
            # Execute the main step
            result = self.execute_step(input_data, config)
            
            # Add execution metadata
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result.update({
                "execution_metadata": {
                    "agent_name": self.agent_name,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "model_id": self.model_id
                }
            })
            
            # Log successful execution
            self._log_info(f"Completed {self.agent_name} execution in {duration:.2f}s")
            
            # Store in execution history
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "success": result.get("success", True),
                "duration": duration,
                "input_data": input_data,
                "result": result
            })
            
            return result
            
        except Exception as e:
            # Handle execution errors
            error_result = self._create_error_result(f"Agent execution failed: {e}")
            
            # Log error
            self._log_error(f"Agent {self.agent_name} failed: {e}")
            
            return error_result
    
    def _validate_inputs(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data and configuration.
        
        Args:
            input_data: Input data to validate
            config: Configuration to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Basic validation - can be overridden by subclasses
            if not isinstance(input_data, dict):
                return {"valid": False, "error": "input_data must be a dictionary"}
            
            if not isinstance(config, dict):
                return {"valid": False, "error": "config must be a dictionary"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}
    
    def _create_success_result(self, output_data: Dict[str, Any], 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized success result.
        
        Args:
            output_data: The main output data
            metadata: Optional metadata to include
            
        Returns:
            Standardized success result dictionary
        """
        return {
            "success": True,
            "agent_name": self.agent_name,
            "output_data": output_data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_error_result(self, error_message: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized error result.
        
        Args:
            error_message: Description of the error
            metadata: Optional metadata to include
            
        Returns:
            Standardized error result dictionary
        """
        return {
            "success": False,
            "agent_name": self.agent_name,
            "error_message": error_message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
    
    def _use_agent_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Use a specific agent tool with error handling.
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        try:
            if self.agent is None:
                return {
                    "error": "Agent not initialized - using fallback mode",
                    "tool_name": tool_name,
                    "fallback_used": True
                }
            
            # Create a query for the agent to execute the tool
            query = self._create_tool_query(tool_name, **kwargs)
            
            # Run the agent
            result = self.agent.run(query)
            
            return {
                "success": True,
                "tool_name": tool_name,
                "result": result,
                "fallback_used": False
            }
            
        except Exception as e:
            return {
                "error": f"Tool execution failed: {e}",
                "tool_name": tool_name,
                "fallback_used": True
            }
    
    def _create_tool_query(self, tool_name: str, **kwargs) -> str:
        """
        Create a natural language query for the agent to execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Natural language query string
        """
        # Convert kwargs to a readable format
        args_str = ", ".join(f"{k}='{v}'" for k, v in kwargs.items())
        
        return f"Please use the {tool_name} tool with the following arguments: {args_str}"
    
    def _log_info(self, message: str):
        """Log an info message."""
        log_entry = {
            "level": "INFO",
            "agent": self.agent_name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[INFO] {self.agent_name}: {message}")
    
    def _log_warning(self, message: str):
        """Log a warning message."""
        log_entry = {
            "level": "WARNING",
            "agent": self.agent_name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[WARNING] {self.agent_name}: {message}")
    
    def _log_error(self, message: str):
        """Log an error message."""
        log_entry = {
            "level": "ERROR",
            "agent": self.agent_name,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        print(f"[ERROR] {self.agent_name}: {message}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history for this agent."""
        return self.execution_history.copy()
    
    def clear_execution_history(self):
        """Clear the execution history."""
        self.execution_history.clear()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_name": self.agent_name,
            "model_id": self.model_id,
            "agent_initialized": self.agent is not None,
            "execution_count": len(self.execution_history),
            "tools_available": len(self._get_agent_tools()) if self._get_agent_tools() else 0
        }