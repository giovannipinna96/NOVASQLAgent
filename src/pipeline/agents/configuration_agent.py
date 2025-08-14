"""Configuration agent for handling pipeline configuration operations.

This agent manages configuration file reading, writing, and validation
for the SQL generation pipeline.
"""

import json
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent

# Import existing LLM components for configuration management
try:
    from model.LLMasJudge import LLMasJudgeSystem
    from ..tools.config_tools import (
        read_config_file,
        write_config_file,
        update_config,
        create_default_config,
        validate_config_file
    )
except ImportError:
    # Fallback imports
    LLMasJudgeSystem = None
    read_config_file = None
    write_config_file = None
    update_config = None
    create_default_config = None
    validate_config_file = None


class ConfigurationAgent(BasePipelineAgent):
    """
    Agent responsible for configuration management in the pipeline.
    
    This agent handles:
    - Reading configuration files
    - Creating default configurations  
    - Updating configuration values
    - Validating configuration settings
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="configuration_agent")
    
    def _get_agent_tools(self) -> List:
        """Get configuration-related tools for the agent."""
        tools = []
        
        if read_config_file:
            tools.append(read_config_file)
        if write_config_file:
            tools.append(write_config_file)
        if update_config:
            tools.append(update_config)
        if create_default_config:
            tools.append(create_default_config)
        if validate_config_file:
            tools.append(validate_config_file)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute configuration step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - config_file_path: Path to configuration file
                - operation: Operation to perform (read, create, update, validate)
                - config_updates: Optional updates to apply
            config: Configuration parameters for this step
            
        Returns:
            Dictionary containing configuration operation results
        """
        try:
            # Extract required parameters
            config_file_path = input_data.get("config_file_path", "./pipeline_config.json")
            operation = input_data.get("operation", "read")
            config_updates = input_data.get("config_updates", {})
            
            # Execute the requested operation
            if operation == "read":
                return self._read_configuration(config_file_path)
            elif operation == "create":
                return self._create_configuration(config_file_path, config)
            elif operation == "update":
                return self._update_configuration(config_file_path, config_updates)
            elif operation == "validate":
                return self._validate_configuration(config_file_path)
            else:
                return self._create_error_result(f"Unknown operation: {operation}")
        
        except Exception as e:
            return self._create_error_result(f"Configuration step failed: {e}")
    
    def _read_configuration(self, config_file_path: str) -> Dict[str, Any]:
        """Read configuration from file."""
        try:
            if read_config_file:
                # Use the tool
                result = self._use_agent_tool("read_config_file", config_path=config_file_path)
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "config_data": tool_result["config_data"],
                            "config_file": tool_result["config_file"],
                            "file_size": tool_result.get("file_size", 0),
                            "last_modified": tool_result.get("last_modified", "")
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "Failed to read config"))
                
                else:
                    # Fallback mode
                    return self._read_config_fallback(config_file_path)
            else:
                return self._read_config_fallback(config_file_path)
                
        except Exception as e:
            return self._create_error_result(f"Failed to read configuration: {e}")
    
    def _create_configuration(self, config_file_path: str, config_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new configuration file."""
        try:
            overwrite = config_params.get("overwrite", False)
            
            if create_default_config:
                # Use the tool
                result = self._use_agent_tool(
                    "create_default_config",
                    config_path=config_file_path,
                    overwrite=overwrite
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "config_file": tool_result["config_file"],
                            "file_size": tool_result.get("file_size", 0),
                            "default_config_created": True,
                            "config_preview": tool_result.get("config_preview", {})
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "Failed to create config"))
                
                else:
                    # Fallback mode
                    return self._create_config_fallback(config_file_path, overwrite)
            else:
                return self._create_config_fallback(config_file_path, overwrite)
                
        except Exception as e:
            return self._create_error_result(f"Failed to create configuration: {e}")
    
    def _update_configuration(self, config_file_path: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration values."""
        try:
            update_results = []
            
            for key_path, new_value in updates.items():
                if update_config:
                    # Use the tool
                    result = self._use_agent_tool(
                        "update_config",
                        config_path=config_file_path,
                        key_path=key_path,
                        new_value=json.dumps(new_value)
                    )
                    
                    if result.get("success", False):
                        tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                        update_results.append(tool_result)
                    else:
                        update_results.append({
                            "success": False,
                            "key_path": key_path,
                            "error": result.get("error", "Update failed")
                        })
                else:
                    # Fallback mode
                    fallback_result = self._update_config_fallback(config_file_path, key_path, new_value)
                    update_results.append(fallback_result)
            
            successful_updates = [r for r in update_results if r.get("success", False)]
            
            return self._create_success_result({
                "updates_applied": len(successful_updates),
                "total_updates": len(updates),
                "update_results": update_results,
                "config_file": config_file_path
            })
            
        except Exception as e:
            return self._create_error_result(f"Failed to update configuration: {e}")
    
    def _validate_configuration(self, config_file_path: str) -> Dict[str, Any]:
        """Validate configuration file."""
        try:
            if validate_config_file:
                # Use the tool
                result = self._use_agent_tool("validate_config_file", config_path=config_file_path)
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    return self._create_success_result({
                        "valid": tool_result.get("valid", False),
                        "validation_issues": tool_result.get("validation_issues", []),
                        "config_summary": tool_result.get("config_summary", {}),
                        "recommendations": tool_result.get("recommendations", [])
                    })
                else:
                    # Fallback mode
                    return self._validate_config_fallback(config_file_path)
            else:
                return self._validate_config_fallback(config_file_path)
                
        except Exception as e:
            return self._create_error_result(f"Failed to validate configuration: {e}")
    
    def _read_config_fallback(self, config_file_path: str) -> Dict[str, Any]:
        """Fallback method for reading configuration."""
        try:
            config_path = Path(config_file_path)
            
            if not config_path.exists():
                return self._create_error_result(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return self._create_success_result({
                "config_data": config_data,
                "config_file": str(config_path),
                "file_size": config_path.stat().st_size,
                "fallback_mode": True
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback config read failed: {e}")
    
    def _create_config_fallback(self, config_file_path: str, overwrite: bool) -> Dict[str, Any]:
        """Fallback method for creating configuration."""
        try:
            config_path = Path(config_file_path)
            
            if config_path.exists() and not overwrite:
                return self._create_error_result(f"Configuration file already exists: {config_path}")
            
            # Create a basic default configuration
            default_config = {
                "pipeline_name": "sql_generation_pipeline",
                "pipeline_version": "1.0.0",
                "processing_mode": "sequential",
                "configuration_step": {
                    "config_file_path": "./pipeline_config.json",
                    "auto_create": True,
                    "backup_existing": True
                },
                "prompt_variant": {
                    "num_variants": 3,
                    "max_variant_length": 500
                },
                "sql_generation": {
                    "include_comments": True,
                    "validate_syntax": True
                }
            }
            
            # Create directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            return self._create_success_result({
                "config_file": str(config_path),
                "file_size": config_path.stat().st_size,
                "default_config_created": True,
                "fallback_mode": True
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback config creation failed: {e}")
    
    def _update_config_fallback(self, config_file_path: str, key_path: str, new_value: Any) -> Dict[str, Any]:
        """Fallback method for updating configuration."""
        try:
            config_path = Path(config_file_path)
            
            if not config_path.exists():
                return {"success": False, "key_path": key_path, "error": "Config file not found"}
            
            # Read existing config
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Navigate and update
            keys = key_path.split('.')
            current = config_data
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            old_value = current.get(keys[-1], "NOT_SET")
            current[keys[-1]] = new_value
            
            # Write back
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
            
            return {
                "success": True,
                "key_path": key_path,
                "old_value": old_value,
                "new_value": new_value,
                "fallback_mode": True
            }
            
        except Exception as e:
            return {"success": False, "key_path": key_path, "error": str(e)}
    
    def _validate_config_fallback(self, config_file_path: str) -> Dict[str, Any]:
        """Fallback method for validating configuration."""
        try:
            config_path = Path(config_file_path)
            
            if not config_path.exists():
                return self._create_error_result(f"Configuration file not found: {config_path}")
            
            # Basic validation
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            issues = []
            
            # Check required fields
            required_fields = ["pipeline_name", "processing_mode"]
            for field in required_fields:
                if field not in config_data:
                    issues.append(f"Missing required field: {field}")
            
            return self._create_success_result({
                "valid": len(issues) == 0,
                "validation_issues": issues,
                "config_summary": {
                    "fields_count": len(config_data),
                    "pipeline_name": config_data.get("pipeline_name", "unknown")
                },
                "fallback_mode": True
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback config validation failed: {e}")