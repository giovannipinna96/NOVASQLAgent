"""SQL translation agent for the SQL generation pipeline.

This agent handles translating SQL queries from one dialect to another
using SQLGlot-based translation capabilities.
"""

import json
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent
from SQLTranlator.sql_translator import SQLTranslator, SQLDialect

try:
    from ..tools.sql_tools import translate_sql_query, validate_sql_syntax
except ImportError:
    # Fallback imports
    translate_sql_query = None
    validate_sql_syntax = None


class TranslationAgent(BasePipelineAgent):
    """
    Agent responsible for SQL dialect translation in the pipeline.
    
    This agent handles:
    - Translating SQL queries between different dialects
    - Validating translated queries
    - Optimizing queries for target dialects
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="translation_agent")
    
    def _get_agent_tools(self) -> List:
        """Get SQL translation tools for the agent."""
        tools = []
        
        if translate_sql_query:
            tools.append(translate_sql_query)
        if validate_sql_syntax:
            tools.append(validate_sql_syntax)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute SQL translation step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - sql_query: The SQL query to translate
                - merged_query: Optional merged query to translate instead
                - source_dialect: Source SQL dialect (optional)
            config: Configuration parameters for this step including:
                - target_dialect: Target SQL dialect for translation
                - preserve_comments: Whether to preserve comments during translation
                - optimize_for_dialect: Whether to optimize for target dialect
                - validate_translation: Whether to validate the translated query
                
        Returns:
            Dictionary containing translated query and metadata
        """
        try:
            # Extract input query
            sql_query = input_data.get("sql_query", "")
            merged_query = input_data.get("merged_query", "")
            source_dialect = input_data.get("source_dialect", "generic")
            
            # Use merged query if available, otherwise use sql_query
            query_to_translate = merged_query if merged_query and merged_query.strip() else sql_query
            
            # Extract configuration
            target_dialect = config.get("target_dialect", None)
            preserve_comments = config.get("preserve_comments", True)
            optimize_for_dialect = config.get("optimize_for_dialect", True)
            validate_translation = config.get("validate_translation", True)
            
            # Check if translation is needed
            if not target_dialect:
                # No target dialect specified, return original query
                return self._create_success_result({
                    "translated_query": query_to_translate,
                    "translation_performed": False,
                    "source_dialect": source_dialect,
                    "target_dialect": "none_specified",
                    "translation_metadata": {
                        "reason": "No target dialect specified",
                        "original_query": query_to_translate
                    }
                })
            
            if not query_to_translate or not query_to_translate.strip():
                return self._create_error_result("No SQL query provided for translation")
            
            # Check if source and target dialects are the same
            if source_dialect.lower() == target_dialect.lower():
                return self._create_success_result({
                    "translated_query": query_to_translate,
                    "translation_performed": False,
                    "source_dialect": source_dialect,
                    "target_dialect": target_dialect,
                    "translation_metadata": {
                        "reason": "Source and target dialects are the same",
                        "original_query": query_to_translate
                    }
                })
            
            # Perform translation
            translation_result = self._translate_query(
                query_to_translate, source_dialect, target_dialect
            )
            
            if not translation_result.get("success", False):
                # Translation failed, return original query with warning
                self._log_warning(f"Translation failed: {translation_result.get('error_message', 'Unknown error')}")
                return self._create_success_result({
                    "translated_query": query_to_translate,
                    "translation_performed": False,
                    "source_dialect": source_dialect,
                    "target_dialect": target_dialect,
                    "translation_error": translation_result.get("error_message", "Translation failed"),
                    "translation_metadata": {
                        "reason": "Translation failed, using original query",
                        "original_query": query_to_translate
                    }
                })
            
            translated_query = translation_result["output_data"]["translated_query"]
            
            # Validate translated query if requested
            validation_result = None
            if validate_translation and translated_query:
                validation_result = self._validate_translated_query(translated_query, target_dialect)
                
                # If validation fails, consider falling back to original
                if validation_result and not validation_result.get("valid", True):
                    self._log_warning(f"Translated query validation failed: {validation_result.get('error_message', 'Validation failed')}")
            
            # Apply dialect-specific optimizations if requested
            optimized_query = translated_query
            optimization_applied = False
            if optimize_for_dialect:
                optimization_result = self._optimize_for_dialect(translated_query, target_dialect)
                if optimization_result.get("success", False):
                    optimized_query = optimization_result["optimized_query"]
                    optimization_applied = True
            
            return self._create_success_result({
                "translated_query": optimized_query,
                "translation_performed": True,
                "source_dialect": source_dialect,
                "target_dialect": target_dialect,
                "optimization_applied": optimization_applied,
                "validation_result": validation_result,
                "translation_metadata": {
                    "original_query": query_to_translate,
                    "original_query_length": len(query_to_translate),
                    "translated_query_length": len(optimized_query),
                    "preserve_comments": preserve_comments,
                    **translation_result["output_data"].get("translation_metadata", {})
                }
            })
        
        except Exception as e:
            return self._create_error_result(f"SQL translation failed: {e}")
    
    def _translate_query(self, sql_query: str, source_dialect: str, target_dialect: str) -> Dict[str, Any]:
        """Translate SQL query from source to target dialect."""
        try:
            if translate_sql_query:
                # Use the tool
                result = self._use_agent_tool(
                    "translate_sql_query",
                    sql_query=sql_query,
                    target_dialect=target_dialect,
                    source_dialect=source_dialect
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "translated_query": tool_result.get("translated_query", ""),
                            "translation_metadata": tool_result.get("translation_metadata", {})
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "Translation tool failed"))
                else:
                    # Fallback mode
                    return self._translate_query_fallback(sql_query, source_dialect, target_dialect)
            else:
                return self._translate_query_fallback(sql_query, source_dialect, target_dialect)
                
        except Exception as e:
            return self._create_error_result(f"Query translation failed: {e}")
    
    def _validate_translated_query(self, translated_query: str, target_dialect: str) -> Optional[Dict[str, Any]]:
        """Validate the translated query."""
        try:
            if validate_sql_syntax:
                # Use the tool
                result = self._use_agent_tool(
                    "validate_sql_syntax",
                    sql_query=translated_query,
                    dialect=target_dialect
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    return tool_result
                else:
                    # Fallback validation
                    return self._validate_query_fallback(translated_query, target_dialect)
            else:
                return self._validate_query_fallback(translated_query, target_dialect)
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {e}",
                "fallback_mode": True
            }
    
    def _optimize_for_dialect(self, translated_query: str, target_dialect: str) -> Dict[str, Any]:
        """Apply dialect-specific optimizations."""
        try:
            optimized_query = translated_query
            optimizations_applied = []
            
            # Apply dialect-specific optimizations
            if target_dialect.lower() == "postgresql":
                optimized_query, postgres_opts = self._optimize_for_postgresql(optimized_query)
                optimizations_applied.extend(postgres_opts)
            elif target_dialect.lower() == "mysql":
                optimized_query, mysql_opts = self._optimize_for_mysql(optimized_query)
                optimizations_applied.extend(mysql_opts)
            elif target_dialect.lower() == "sqlite":
                optimized_query, sqlite_opts = self._optimize_for_sqlite(optimized_query)
                optimizations_applied.extend(sqlite_opts)
            elif target_dialect.lower() == "bigquery":
                optimized_query, bigquery_opts = self._optimize_for_bigquery(optimized_query)
                optimizations_applied.extend(bigquery_opts)
            
            return {
                "success": True,
                "optimized_query": optimized_query,
                "optimizations_applied": optimizations_applied
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Optimization failed: {e}",
                "optimized_query": translated_query
            }
    
    def _translate_query_fallback(self, sql_query: str, source_dialect: str, target_dialect: str) -> Dict[str, Any]:
        """Fallback method for query translation."""
        try:
            # Simple fallback - apply basic dialect-specific transformations
            translated_query = sql_query
            transformations = []
            
            # PostgreSQL specific transformations
            if target_dialect.lower() == "postgresql":
                # Replace LIMIT syntax variations
                if "TOP " in translated_query.upper():
                    # SQL Server TOP to PostgreSQL LIMIT
                    import re
                    translated_query = re.sub(r'SELECT\s+TOP\s+(\d+)', r'SELECT', translated_query, flags=re.IGNORECASE)
                    translated_query = translated_query + f" LIMIT {re.search(r'TOP\\s+(\\d+)', sql_query, re.IGNORECASE).group(1)}"
                    transformations.append("Converted TOP to LIMIT")
            
            # MySQL specific transformations
            elif target_dialect.lower() == "mysql":
                # Add backticks for identifiers if needed
                if "ORDER BY" in translated_query.upper():
                    transformations.append("MySQL dialect formatting applied")
            
            # BigQuery specific transformations
            elif target_dialect.lower() == "bigquery":
                # Replace double quotes with backticks
                translated_query = translated_query.replace('"', '`')
                if '"' in sql_query:
                    transformations.append("Replaced quotes with backticks for BigQuery")
            
            return self._create_success_result({
                "translated_query": translated_query,
                "translation_metadata": {
                    "fallback_mode": True,
                    "transformations_applied": transformations,
                    "source_dialect": source_dialect,
                    "target_dialect": target_dialect
                }
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback translation failed: {e}")
    
    def _validate_query_fallback(self, query: str, dialect: str) -> Dict[str, Any]:
        """Fallback method for query validation."""
        try:
            # Basic validation
            sql_lower = query.lower().strip()
            
            # Check basic SQL structure
            valid_starts = ["select", "insert", "update", "delete", "create", "drop", "alter", "with"]
            starts_correctly = any(sql_lower.startswith(start) for start in valid_starts)
            
            # Check for balanced parentheses
            paren_count = query.count('(') - query.count(')')
            balanced_parens = paren_count == 0
            
            # Dialect-specific checks
            dialect_issues = []
            if dialect.lower() == "bigquery" and '"' in query:
                dialect_issues.append("BigQuery prefers backticks over double quotes")
            
            is_valid = starts_correctly and balanced_parens
            
            return {
                "valid": is_valid,
                "dialect_issues": dialect_issues,
                "basic_checks": {
                    "starts_correctly": starts_correctly,
                    "balanced_parentheses": balanced_parens
                },
                "fallback_mode": True
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Fallback validation failed: {e}",
                "fallback_mode": True
            }
    
    def _optimize_for_postgresql(self, query: str) -> tuple:
        """Apply PostgreSQL-specific optimizations."""
        optimizations = []
        optimized_query = query
        
        # Add EXPLAIN ANALYZE comment for performance testing
        if not query.strip().startswith("--"):
            optimized_query = "-- Use EXPLAIN ANALYZE to test performance\\n" + optimized_query
            optimizations.append("Added performance testing comment")
        
        return optimized_query, optimizations
    
    def _optimize_for_mysql(self, query: str) -> tuple:
        """Apply MySQL-specific optimizations."""
        optimizations = []
        optimized_query = query
        
        # Ensure backticks for reserved words
        if " limit " in query.lower() and not query.endswith(";"):
            optimized_query = optimized_query + ";"
            optimizations.append("Added semicolon terminator")
        
        return optimized_query, optimizations
    
    def _optimize_for_sqlite(self, query: str) -> tuple:
        """Apply SQLite-specific optimizations."""
        optimizations = []
        optimized_query = query
        
        # SQLite is generally simple, minimal optimizations needed
        if "AUTOINCREMENT" in query.upper():
            optimizations.append("SQLite AUTOINCREMENT detected")
        
        return optimized_query, optimizations
    
    def _optimize_for_bigquery(self, query: str) -> tuple:
        """Apply BigQuery-specific optimizations."""
        optimizations = []
        optimized_query = query
        
        # Replace double quotes with backticks
        if '"' in optimized_query:
            optimized_query = optimized_query.replace('"', '`')
            optimizations.append("Replaced quotes with backticks")
        
        # Add dataset reference comment
        if "FROM " in optimized_query.upper() and "project." not in optimized_query.lower():
            optimized_query = "-- Consider fully qualified table names: project.dataset.table\\n" + optimized_query
            optimizations.append("Added dataset reference guidance")
        
        return optimized_query, optimizations