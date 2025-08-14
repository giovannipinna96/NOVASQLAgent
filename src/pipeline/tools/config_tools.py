"""Configuration management tools for the SQL generation pipeline.

This module provides tools that can be used by LLM agents to read, write,
and manage configuration files for the pipeline.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from smolagents import tool

from ..models.pipeline_config import PipelineConfig


@tool
def read_config_file(config_path: str) -> str:
    """
    Read a pipeline configuration file and return its contents.
    
    Args:
        config_path: Path to the configuration file to read
        
    Returns:
        JSON string containing the configuration data
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            return json.dumps({
                "error": f"Configuration file not found: {config_path}",
                "suggestion": "Use create_default_config to create a new configuration file"
            })
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        return json.dumps({
            "success": True,
            "config_file": str(config_path),
            "config_data": config_data,
            "file_size": config_path.stat().st_size,
            "last_modified": datetime.fromtimestamp(config_path.stat().st_mtime).isoformat()
        })
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid JSON in configuration file: {e}",
            "file": str(config_path)
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to read configuration file: {e}",
            "file": str(config_path)
        })


@tool
def write_config_file(config_path: str, config_data: str, backup_existing: bool = True) -> str:
    """
    Write configuration data to a file.
    
    Args:
        config_path: Path where to write the configuration file
        config_data: JSON string containing the configuration data
        backup_existing: Whether to backup existing file before overwriting
        
    Returns:
        JSON string with operation result
    """
    try:
        config_path = Path(config_path)
        
        # Parse config data to validate JSON
        try:
            config_dict = json.loads(config_data)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid JSON data provided: {e}"
            })
        
        # Create backup if file exists and backup is requested
        backup_path = None
        if config_path.exists() and backup_existing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = config_path.with_suffix(f".backup_{timestamp}.json")
            shutil.copy2(config_path, backup_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "config_file": str(config_path),
            "backup_created": str(backup_path) if backup_path else None,
            "file_size": config_path.stat().st_size,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to write configuration file: {e}",
            "file": str(config_path)
        })


@tool
def update_config(config_path: str, key_path: str, new_value: str) -> str:
    """
    Update a specific configuration value using dot notation.
    
    Args:
        config_path: Path to the configuration file
        key_path: Dot-separated path to the configuration key (e.g., "sql_generation.max_tokens")
        new_value: New value to set (as JSON string)
        
    Returns:
        JSON string with operation result
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            return json.dumps({
                "error": f"Configuration file not found: {config_path}",
                "suggestion": "Use create_default_config to create a new configuration file"
            })
        
        # Read existing config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Parse new value
        try:
            new_value_parsed = json.loads(new_value)
        except json.JSONDecodeError:
            # If not valid JSON, treat as string
            new_value_parsed = new_value
        
        # Navigate to the key using dot notation
        keys = key_path.split('.')
        current = config_data
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Store old value
        final_key = keys[-1]
        old_value = current.get(final_key, "NOT_SET")
        
        # Set new value
        current[final_key] = new_value_parsed
        
        # Create backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f".backup_{timestamp}.json")
        shutil.copy2(config_path, backup_path)
        
        # Write updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "config_file": str(config_path),
            "key_path": key_path,
            "old_value": old_value,
            "new_value": new_value_parsed,
            "backup_created": str(backup_path),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to update configuration: {e}",
            "file": str(config_path),
            "key_path": key_path
        })


@tool
def create_default_config(config_path: str, overwrite: bool = False) -> str:
    """
    Create a default pipeline configuration file.
    
    Args:
        config_path: Path where to create the configuration file
        overwrite: Whether to overwrite existing file
        
    Returns:
        JSON string with operation result
    """
    try:
        config_path = Path(config_path)
        
        if config_path.exists() and not overwrite:
            return json.dumps({
                "error": f"Configuration file already exists: {config_path}",
                "suggestion": "Use overwrite=true to replace existing file"
            })
        
        # Create default configuration
        default_config = PipelineConfig()
        config_dict = default_config.to_dict()
        
        # Create directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "config_file": str(config_path),
            "file_size": config_path.stat().st_size,
            "default_config_created": True,
            "timestamp": datetime.now().isoformat(),
            "config_preview": {
                "pipeline_name": config_dict["pipeline_name"],
                "pipeline_version": config_dict["pipeline_version"],
                "processing_mode": config_dict["processing_mode"],
                "step_count": len([k for k in config_dict.keys() if k.endswith("_step") or k in ["prompt_variant", "relevance_evaluation", "sql_generation", "query_merge", "translation", "output"]])
            }
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to create default configuration: {e}",
            "file": str(config_path)
        })


@tool
def validate_config_file(config_path: str) -> str:
    """
    Validate a pipeline configuration file for correctness.
    
    Args:
        config_path: Path to the configuration file to validate
        
    Returns:
        JSON string with validation results
    """
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            return json.dumps({
                "valid": False,
                "error": f"Configuration file not found: {config_path}"
            })
        
        # Read and parse config
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Try to create PipelineConfig from data
        try:
            config = PipelineConfig.from_dict(config_data)
            validation_issues = config.validate()
            
            return json.dumps({
                "valid": len(validation_issues) == 0,
                "config_file": str(config_path),
                "validation_issues": validation_issues,
                "config_summary": {
                    "pipeline_name": config.pipeline_name,
                    "pipeline_version": config.pipeline_version,
                    "processing_mode": config.processing_mode.value,
                    "steps_configured": len([
                        config.configuration_step,
                        config.prompt_variant,
                        config.relevance_evaluation,
                        config.sql_generation,
                        config.query_merge,
                        config.translation,
                        config.output
                    ])
                },
                "recommendations": _get_config_recommendations(config),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": f"Configuration validation failed: {e}",
                "config_file": str(config_path)
            })
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "valid": False,
            "error": f"Invalid JSON in configuration file: {e}",
            "config_file": str(config_path)
        })
    except Exception as e:
        return json.dumps({
            "valid": False,
            "error": f"Failed to validate configuration file: {e}",
            "config_file": str(config_path)
        })


def _get_config_recommendations(config: PipelineConfig) -> list:
    """Get configuration recommendations based on settings."""
    recommendations = []
    
    if config.prompt_variant.num_variants < 3:
        recommendations.append("Consider increasing num_variants to at least 3 for better prompt diversity")
    
    if config.relevance_evaluation.confidence_threshold > 0.9:
        recommendations.append("High confidence threshold may reject relevant descriptions")
    
    if config.sql_generation.max_query_length < 1000:
        recommendations.append("Consider increasing max_query_length for complex queries")
    
    if not config.sql_generation.validate_syntax:
        recommendations.append("Enable SQL syntax validation for better output quality")
    
    if not config.output.backup_existing:
        recommendations.append("Enable backup_existing to prevent data loss")
    
    return recommendations