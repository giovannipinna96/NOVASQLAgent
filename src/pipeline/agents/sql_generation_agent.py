"""SQL generation agent for the SQL generation pipeline.

This agent handles generating SQL queries from the original prompt 
and relevant descriptions using LLM SQL generation capabilities.
"""

import json
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent
from model.LLMsql import SQLLLMGenerator, SQLGenerationConfig, SQLDialect, SchemaInfo

try:
    from ..tools.sql_tools import (
        generate_sql_query,
        validate_sql_syntax
    )
except ImportError:
    # Fallback imports
    generate_sql_query = None
    validate_sql_syntax = None


class SQLGenerationAgent(BasePipelineAgent):
    """
    Agent responsible for SQL query generation in the pipeline.
    
    This agent handles:
    - Generating SQL queries from prompts and descriptions
    - Validating generated SQL syntax
    - Providing metadata about generated queries
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="sql_generation_agent")
    
    def _get_agent_tools(self) -> List:
        """Get SQL generation tools for the agent."""
        tools = []
        
        if generate_sql_query:
            tools.append(generate_sql_query)
        if validate_sql_syntax:
            tools.append(validate_sql_syntax)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SQL generation step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - original_prompt: The original prompt for SQL generation
                - relevant_descriptions: List of relevant descriptions
                - prompt_variants: Optional list of prompt variants for context
            config: Configuration parameters for this step including:
                - include_comments: Whether to include comments in SQL
                - format_output: Whether to format the SQL output
                - validate_syntax: Whether to validate SQL syntax
                - max_query_length: Maximum length for generated query
                - target_dialect: SQL dialect to generate for
                
        Returns:
            Dictionary containing generated SQL queries and metadata
        """
        try:
            # Extract required parameters
            original_prompt = input_data.get("original_prompt", "")
            relevant_descriptions = input_data.get("relevant_descriptions", [])
            prompt_variants = input_data.get("prompt_variants", [])
            
            # Extract configuration
            include_comments = config.get("include_comments", True)
            format_output = config.get("format_output", True)
            validate_syntax = config.get("validate_syntax", True)
            max_query_length = config.get("max_query_length", 2000)
            target_dialect = config.get("target_dialect", "postgresql")
            
            # Validate inputs
            if not original_prompt.strip():
                return self._create_error_result("Original prompt cannot be empty")
            
            # Generate SQL query
            generation_result = self._generate_sql_query(
                original_prompt,
                relevant_descriptions,
                target_dialect,
                include_comments,
                validate_syntax
            )
            
            if not generation_result.get("success", False):
                return generation_result
            
            # Extract generated query
            sql_query = generation_result["output_data"]["sql_query"]
            
            # Check query length
            if sql_query and len(sql_query) > max_query_length:
                self._log_warning(f"Generated query exceeds max length ({len(sql_query)} > {max_query_length})")
                # Optionally truncate or request regeneration
                if len(sql_query) > max_query_length * 1.5:  # Too long, truncate
                    sql_query = sql_query[:max_query_length] + "\\n-- Query truncated due to length limit"
            
            # Perform additional validation if enabled
            validation_result = None
            if validate_syntax and sql_query:
                validation_result = self._validate_generated_sql(sql_query, target_dialect)
            
            # Generate multiple queries if needed (experimental)
            alternative_queries = []
            if len(prompt_variants) > 1:
                alternative_queries = self._generate_alternative_queries(
                    prompt_variants, relevant_descriptions, target_dialect, include_comments
                )
            
            return self._create_success_result({
                "primary_query": sql_query,
                "alternative_queries": alternative_queries,
                "validation_result": validation_result,
                "generation_metadata": {
                    "original_prompt": original_prompt,
                    "relevant_descriptions_count": len(relevant_descriptions),
                    "target_dialect": target_dialect,
                    "query_length": len(sql_query) if sql_query else 0,
                    "include_comments": include_comments,
                    "syntax_validated": bool(validation_result),
                    **generation_result["output_data"].get("generation_metadata", {})
                }
            })
        
        except Exception as e:
            return self._create_error_result(f"SQL generation failed: {e}")
    
    def _generate_sql_query(self, original_prompt: str, relevant_descriptions: List[str], 
                          dialect: str, include_comments: bool, validate_syntax: bool) -> Dict[str, Any]:
        """Generate SQL query using the original prompt and relevant descriptions."""
        try:
            # Prepare relevant descriptions as JSON string
            descriptions_json = json.dumps(relevant_descriptions)
            
            if generate_sql_query:
                # Use the tool
                result = self._use_agent_tool(
                    "generate_sql_query",
                    prompt=original_prompt,
                    relevant_descriptions=descriptions_json,
                    dialect=dialect,
                    include_comments=include_comments,
                    validate_syntax=validate_syntax
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "sql_query": tool_result.get("sql_query", ""),
                            "syntax_valid": tool_result.get("syntax_valid", True),
                            "syntax_error": tool_result.get("syntax_error", None),
                            "generation_metadata": tool_result.get("generation_metadata", {})
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "SQL generation failed"))
                else:
                    # Fallback mode
                    return self._generate_sql_fallback(original_prompt, relevant_descriptions, dialect)
            else:
                return self._generate_sql_fallback(original_prompt, relevant_descriptions, dialect)
                
        except Exception as e:
            return self._create_error_result(f"SQL query generation failed: {e}")
    
    def _validate_generated_sql(self, sql_query: str, dialect: str) -> Dict[str, Any]:
        """Validate the generated SQL query."""
        try:
            if validate_sql_syntax:
                # Use the tool
                result = self._use_agent_tool(
                    "validate_sql_syntax",
                    sql_query=sql_query,
                    dialect=dialect
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    return tool_result
                else:
                    # Fallback validation
                    return self._validate_sql_fallback(sql_query, dialect)
            else:
                return self._validate_sql_fallback(sql_query, dialect)
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {e}",
                "fallback_mode": True
            }
    
    def _generate_alternative_queries(self, prompt_variants: List[Dict], relevant_descriptions: List[str], 
                                    dialect: str, include_comments: bool) -> List[Dict[str, Any]]:
        """Generate alternative queries using prompt variants."""
        alternative_queries = []
        
        # Generate queries for top variants (limit to avoid too many)
        max_alternatives = min(len(prompt_variants), 3)
        
        for i, variant in enumerate(prompt_variants[:max_alternatives]):
            try:
                variant_text = variant.get("text", "")
                if not variant_text.strip():
                    continue
                
                # Generate SQL for this variant
                generation_result = self._generate_sql_query(
                    variant_text, relevant_descriptions, dialect, include_comments, False
                )
                
                if generation_result.get("success", False):
                    sql_query = generation_result["output_data"]["sql_query"]
                    
                    alternative_queries.append({
                        "variant_id": variant.get("variant_id", i + 1),
                        "variant_text": variant_text,
                        "sql_query": sql_query,
                        "query_length": len(sql_query) if sql_query else 0,
                        "generation_successful": bool(sql_query)
                    })
                else:
                    self._log_warning(f"Failed to generate SQL for variant {i+1}: {generation_result.get('error_message', 'Unknown error')}")
            
            except Exception as e:
                self._log_warning(f"Error generating alternative query for variant {i+1}: {e}")
        
        return alternative_queries
    
    def _generate_sql_fallback(self, original_prompt: str, relevant_descriptions: List[str], dialect: str) -> Dict[str, Any]:
        """Fallback method for SQL generation."""
        try:
            # Simple template-based SQL generation
            prompt_lower = original_prompt.lower()
            
            # Try to identify query type
            if "select" in prompt_lower or "find" in prompt_lower or "get" in prompt_lower:
                # SELECT query
                sql_query = "SELECT * FROM table_name WHERE condition = 'value';"
                if "count" in prompt_lower:
                    sql_query = "SELECT COUNT(*) FROM table_name WHERE condition = 'value';"
            elif "insert" in prompt_lower or "add" in prompt_lower or "create" in prompt_lower:
                # INSERT query
                sql_query = "INSERT INTO table_name (column1, column2) VALUES ('value1', 'value2');"
            elif "update" in prompt_lower or "modify" in prompt_lower or "change" in prompt_lower:
                # UPDATE query
                sql_query = "UPDATE table_name SET column1 = 'new_value' WHERE condition = 'value';"
            elif "delete" in prompt_lower or "remove" in prompt_lower:
                # DELETE query
                sql_query = "DELETE FROM table_name WHERE condition = 'value';"
            else:
                # Default to SELECT
                sql_query = "SELECT * FROM table_name;"
            
            # Add comments with context
            if relevant_descriptions:
                comments = "-- Generated based on: " + "; ".join(relevant_descriptions[:2])
                sql_query = comments + "\\n" + sql_query
            
            return self._create_success_result({
                "sql_query": sql_query,
                "syntax_valid": True,
                "generation_metadata": {
                    "fallback_mode": True,
                    "template_used": True,
                    "relevant_descriptions_used": len(relevant_descriptions)
                }
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback SQL generation failed: {e}")
    
    def _validate_sql_fallback(self, sql_query: str, dialect: str) -> Dict[str, Any]:
        """Fallback method for SQL validation."""
        try:
            # Basic syntax validation
            sql_lower = sql_query.lower().strip()
            
            # Check for basic SQL structure
            valid_starts = ["select", "insert", "update", "delete", "create", "drop", "alter"]
            starts_correctly = any(sql_lower.startswith(start) for start in valid_starts)
            
            # Check for balanced parentheses
            paren_count = sql_query.count('(') - sql_query.count(')')
            balanced_parens = paren_count == 0
            
            # Check for semicolon
            has_semicolon = sql_query.strip().endswith(';')
            
            is_valid = starts_correctly and balanced_parens
            
            issues = []
            if not starts_correctly:
                issues.append("Query does not start with a valid SQL keyword")
            if not balanced_parens:
                issues.append("Unbalanced parentheses")
            if not has_semicolon:
                issues.append("Missing semicolon at end")
            
            return {
                "valid": is_valid,
                "error_message": "; ".join(issues) if issues else None,
                "query_analysis": {
                    "starts_correctly": starts_correctly,
                    "balanced_parentheses": balanced_parens,
                    "has_semicolon": has_semicolon,
                    "query_length": len(sql_query)
                },
                "fallback_mode": True
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Fallback validation failed: {e}",
                "fallback_mode": True
            }