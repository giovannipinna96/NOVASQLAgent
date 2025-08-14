"""SQL generation and processing tools for the pipeline.

This module provides tools for SQL generation, query merging, translation,
and validation using existing LLM components.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from smolagents import tool

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from model.LLMsql import SQLLLMGenerator, SQLGenerationConfig, SQLDialect as SQLModelDialect, SchemaInfo
    from model.LLMmerge import LLMSQLMergerSystem
    from SQLTranlator.sql_translator import SQLTranslator, SQLDialect as TranslatorDialect
except ImportError as e:
    # Fallback for when modules are not available
    SQLLLMGenerator = None
    LLMSQLMergerSystem = None
    SQLTranslator = None
    SQLModelDialect = None
    TranslatorDialect = None


@tool
def generate_sql_query(prompt: str, relevant_descriptions: str = "", dialect: str = "postgresql", 
                      include_comments: bool = True, validate_syntax: bool = True) -> str:
    """
    Generate SQL query from prompt and relevant descriptions.
    
    Args:
        prompt: The original prompt for SQL generation
        relevant_descriptions: JSON string of relevant descriptions to include
        dialect: Target SQL dialect (default: postgresql)
        include_comments: Whether to include comments in generated SQL
        validate_syntax: Whether to validate SQL syntax
        
    Returns:
        JSON string containing generated SQL query and metadata
    """
    try:
        if not prompt.strip():
            return json.dumps({
                "error": "Prompt cannot be empty",
                "sql_query": None
            })
        
        if SQLLLMGenerator is None:
            return json.dumps({
                "error": "SQLLLMGenerator not available - check model imports",
                "sql_query": None
            })
        
        # Parse relevant descriptions
        descriptions_list = []
        if relevant_descriptions:
            try:
                descriptions_data = json.loads(relevant_descriptions)
                if isinstance(descriptions_data, list):
                    descriptions_list = descriptions_data
                elif isinstance(descriptions_data, dict) and "descriptions" in descriptions_data:
                    descriptions_list = descriptions_data["descriptions"]
            except json.JSONDecodeError:
                # Treat as single description if not valid JSON
                descriptions_list = [relevant_descriptions]
        
        # Prepare context for SQL generation
        context = prompt
        if descriptions_list:
            context += "\\n\\nRelevant context:\\n" + "\\n".join(f"- {desc}" for desc in descriptions_list)
        
        try:
            # Create configuration for SQL generation
            sql_config = SQLGenerationConfig(
                model_name="microsoft/phi-4-mini-instruct",
                max_new_tokens=512 if include_comments else 256,
                temperature=0.1
            )
            
            # Map dialect string to SQLDialect enum
            try:
                sql_dialect = SQLModelDialect(dialect.lower())
            except ValueError:
                sql_dialect = SQLModelDialect.POSTGRESQL  # Default fallback
            
            # Initialize SQL generator with existing implementation
            sql_generator = SQLLLMGenerator(config=sql_config, dialect=sql_dialect)
            
            # Generate SQL query using existing method
            generation_result = sql_generator.generate_sql(user_request=context)
            
            # Parse the result based on existing format
            if isinstance(generation_result, dict) and generation_result.get("success", False):
                sql_query = generation_result.get("generated_sql", "")
                metadata = {
                    "query_type": generation_result.get("query_type", "unknown"),
                    "dialect": generation_result.get("dialect", dialect),
                    "raw_output": generation_result.get("raw_output", "")
                }
            else:
                return json.dumps({
                    "error": generation_result.get("error", "SQL generation failed"),
                    "sql_query": None
                })
            
            # Validate syntax if requested
            syntax_valid = True
            syntax_error = None
            if validate_syntax and sql_query:
                try:
                    # Basic syntax validation
                    syntax_valid, syntax_error = _validate_sql_syntax(sql_query, dialect)
                except Exception as validation_error:
                    syntax_error = str(validation_error)
                    syntax_valid = False
            
            return json.dumps({
                "success": True,
                "sql_query": sql_query,
                "dialect": dialect,
                "syntax_valid": syntax_valid,
                "syntax_error": syntax_error,
                "generation_metadata": {
                    "original_prompt": prompt,
                    "descriptions_used": descriptions_list,
                    "context_length": len(context),
                    "query_length": len(sql_query) if sql_query else 0,
                    "include_comments": include_comments,
                    "generation_time": datetime.now().isoformat(),
                    **metadata
                }
            })
            
        except Exception as generation_error:
            return json.dumps({
                "error": f"SQL generation failed: {generation_error}",
                "sql_query": None
            })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to generate SQL query: {e}",
            "sql_query": None
        })


@tool
def merge_sql_queries(queries: str, merge_strategy: str = "union") -> str:
    """
    Merge multiple SQL queries using specified strategy.
    
    Args:
        queries: JSON string containing list of SQL queries to merge
        merge_strategy: Strategy for merging (union, join, cte, subquery)
        
    Returns:
        JSON string containing merged query and metadata
    """
    try:
        # Parse queries list
        try:
            queries_data = json.loads(queries)
            if isinstance(queries_data, list):
                queries_list = queries_data
            elif isinstance(queries_data, dict) and "queries" in queries_data:
                queries_list = queries_data["queries"]
            else:
                return json.dumps({
                    "error": "Invalid queries format - expected list of SQL queries",
                    "merged_query": None
                })
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid JSON format for queries",
                "merged_query": None
            })
        
        if not queries_list:
            return json.dumps({
                "error": "No queries provided for merging",
                "merged_query": None
            })
        
        if len(queries_list) == 1:
            return json.dumps({
                "success": True,
                "merged_query": queries_list[0],
                "merge_strategy": "single_query",
                "merge_metadata": {
                    "original_queries_count": 1,
                    "merge_time": datetime.now().isoformat()
                }
            })
        
        if LLMSQLMergerSystem is None:
            return json.dumps({
                "error": "LLMSQLMergerSystem not available - check model imports",
                "merged_query": None
            })
        
        try:
            # Initialize query merger system with existing implementation
            merger_system = LLMSQLMergerSystem()
            
            # Prepare input text with queries
            input_text = "\\n\\n".join(f"Query {i+1}:\\n{query}" for i, query in enumerate(queries_list))
            
            # Merge queries using existing method
            merge_result = merger_system.merge_queries(input_text, detailed=True)
            
            # Parse merger result based on existing format
            if isinstance(merge_result, dict) and merge_result.get("success", False):
                merged_query = merge_result.get("merged_query", "")
                merge_metadata = {
                    "extracted_queries": merge_result.get("extracted_queries", []),
                    "merge_strategy": merge_result.get("merge_strategy", merge_strategy),
                    "is_valid": merge_result.get("is_valid", None),
                    "validation_error": merge_result.get("validation_error", None)
                }
                merge_successful = bool(merged_query)
            else:
                return json.dumps({
                    "error": merge_result.get("error", "Query merging failed"),
                    "merged_query": None
                })
            
            return json.dumps({
                "success": merge_successful,
                "merged_query": merged_query,
                "merge_strategy": merge_strategy,
                "merge_metadata": {
                    "original_queries_count": len(queries_list),
                    "original_queries": queries_list,
                    "merged_query_length": len(merged_query) if merged_query else 0,
                    "merge_time": datetime.now().isoformat(),
                    **merge_metadata
                }
            })
            
        except Exception as merge_error:
            return json.dumps({
                "error": f"Query merging failed: {merge_error}",
                "merged_query": None
            })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to merge SQL queries: {e}",
            "merged_query": None
        })


@tool
def translate_sql_query(sql_query: str, target_dialect: str, source_dialect: str = "generic") -> str:
    """
    Translate SQL query from source dialect to target dialect.
    
    Args:
        sql_query: The SQL query to translate
        target_dialect: Target SQL dialect to translate to
        source_dialect: Source SQL dialect (default: generic)
        
    Returns:
        JSON string containing translated query and metadata
    """
    try:
        if not sql_query.strip():
            return json.dumps({
                "error": "SQL query cannot be empty",
                "translated_query": None
            })
        
        if SQLTranslator is None or TranslatorDialect is None:
            return json.dumps({
                "error": "SQLTranslator not available - check imports",
                "translated_query": None
            })
        
        try:
            # Map dialect strings to TranslatorDialect enums
            try:
                source_dialect_enum = TranslatorDialect.from_string(source_dialect) if source_dialect != "generic" else None
                target_dialect_enum = TranslatorDialect.from_string(target_dialect)
            except ValueError as e:
                return json.dumps({
                    "error": f"Unsupported dialect: {e}",
                    "translated_query": None
                })
            
            # Initialize SQL translator with existing implementation
            translator = SQLTranslator(source_dialect=source_dialect_enum)
            
            # Perform translation using existing method
            translation_result = translator.translate(
                sql_query=sql_query,
                target_dialect=target_dialect_enum,
                pretty=True
            )
            
            # Parse translation result based on existing TranslationResult format
            if hasattr(translation_result, 'translated_sql'):
                translated_query = translation_result.translated_sql
                translation_success = bool(translated_query)
                translation_metadata = {
                    "source_dialect": translation_result.source_dialect,
                    "target_dialect": translation_result.target_dialect,
                    "warnings": translation_result.warnings
                }
                translation_error = None
            else:
                translated_query = str(translation_result)
                translation_success = bool(translated_query)
                translation_metadata = {}
                translation_error = None
            
            return json.dumps({
                "success": translation_success,
                "translated_query": translated_query,
                "source_dialect": source_dialect,
                "target_dialect": target_dialect,
                "translation_error": translation_error,
                "translation_metadata": {
                    "original_query": sql_query,
                    "original_query_length": len(sql_query),
                    "translated_query_length": len(translated_query) if translated_query else 0,
                    "translation_time": datetime.now().isoformat(),
                    **translation_metadata
                }
            })
            
        except Exception as translation_error:
            return json.dumps({
                "error": f"SQL translation failed: {translation_error}",
                "translated_query": None
            })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to translate SQL query: {e}",
            "translated_query": None
        })


@tool
def validate_sql_syntax(sql_query: str, dialect: str = "generic") -> str:
    """
    Validate SQL query syntax for specified dialect.
    
    Args:
        sql_query: The SQL query to validate
        dialect: SQL dialect to validate against (default: generic)
        
    Returns:
        JSON string containing validation results
    """
    try:
        if not sql_query.strip():
            return json.dumps({
                "valid": False,
                "error": "SQL query cannot be empty"
            })
        
        # Perform syntax validation
        is_valid, error_message = _validate_sql_syntax(sql_query, dialect)
        
        # Additional analysis
        query_analysis = _analyze_sql_query(sql_query)
        
        return json.dumps({
            "valid": is_valid,
            "error_message": error_message,
            "dialect": dialect,
            "query_analysis": query_analysis,
            "validation_metadata": {
                "query_length": len(sql_query),
                "validation_time": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return json.dumps({
            "valid": False,
            "error": f"Validation failed: {e}"
        })


def _validate_sql_syntax(sql_query: str, dialect: str) -> tuple:
    """Validate SQL syntax using existing SQLTranslator validation."""
    try:
        # Use existing SQLTranslator validation if available
        if SQLTranslator and TranslatorDialect:
            try:
                # Map dialect string to enum
                try:
                    dialect_enum = TranslatorDialect.from_string(dialect)
                except ValueError:
                    dialect_enum = None
                
                # Use existing validate_sql method
                validation_result = SQLTranslator.validate_sql(sql_query, dialect_enum)
                
                if validation_result.get("is_valid", False):
                    return True, None
                else:
                    errors = validation_result.get("errors", [])
                    return False, "; ".join(errors) if errors else "Syntax validation failed"
                    
            except Exception as e:
                return False, str(e)
        
        # Basic validation fallback
        sql_lower = sql_query.lower().strip()
        
        # Check for basic SQL structure
        if not any(sql_lower.startswith(kw) for kw in ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']):
            return False, "Query must start with a valid SQL keyword"
        
        # Check for balanced parentheses
        paren_count = sql_query.count('(') - sql_query.count(')')
        if paren_count != 0:
            return False, "Unbalanced parentheses in query"
        
        # Check for basic semicolon issues
        if sql_query.count(';') > 1:
            return False, "Multiple statements detected"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {e}"


def _analyze_sql_query(sql_query: str) -> Dict[str, Any]:
    """Analyze SQL query structure and components."""
    try:
        query_lower = sql_query.lower()
        
        # Identify query type
        query_types = {
            'SELECT': 'select' in query_lower,
            'INSERT': 'insert' in query_lower,
            'UPDATE': 'update' in query_lower,
            'DELETE': 'delete' in query_lower,
            'CREATE': 'create' in query_lower,
            'DROP': 'drop' in query_lower,
            'ALTER': 'alter' in query_lower
        }
        
        primary_type = next((qtype for qtype, present in query_types.items() if present), 'UNKNOWN')
        
        # Count various components
        components = {
            'tables_referenced': len([word for word in sql_query.split() if word.lower() in ['from', 'join', 'into', 'update']]),
            'conditions': query_lower.count('where'),
            'joins': query_lower.count('join'),
            'subqueries': query_lower.count('select') - 1 if primary_type == 'SELECT' else query_lower.count('select'),
            'functions': len([f for f in ['count(', 'sum(', 'avg(', 'max(', 'min('] if f in query_lower]),
            'group_by': 'group by' in query_lower,
            'order_by': 'order by' in query_lower,
            'having': 'having' in query_lower
        }
        
        # Complexity estimation
        complexity_score = (
            components['joins'] * 2 +
            components['subqueries'] * 3 +
            components['functions'] +
            (2 if components['group_by'] else 0) +
            (1 if components['having'] else 0)
        )
        
        complexity_level = "low" if complexity_score <= 2 else "medium" if complexity_score <= 6 else "high"
        
        return {
            "query_type": primary_type,
            "complexity_level": complexity_level,
            "complexity_score": complexity_score,
            "components": components,
            "estimated_performance": "good" if complexity_score <= 4 else "moderate" if complexity_score <= 8 else "review_needed"
        }
        
    except Exception:
        return {
            "query_type": "UNKNOWN",
            "complexity_level": "unknown",
            "analysis_error": "Failed to analyze query structure"
        }