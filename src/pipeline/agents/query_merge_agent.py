"""Query merge agent for the SQL generation pipeline.

This agent handles merging multiple SQL queries into a single cohesive query
using various merging strategies.
"""

import json
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent
from model.LLMmerge import LLMSQLMergerSystem

try:
    from ..tools.sql_tools import merge_sql_queries
except ImportError:
    # Fallback imports
    merge_sql_queries = None


class QueryMergeAgent(BasePipelineAgent):
    """
    Agent responsible for merging SQL queries in the pipeline.
    
    This agent handles:
    - Merging multiple SQL queries using different strategies
    - Validating merged queries
    - Optimizing query structure
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="query_merge_agent")
    
    def _get_agent_tools(self) -> List:
        """Get query merging tools for the agent."""
        tools = []
        
        if merge_sql_queries:
            tools.append(merge_sql_queries)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query merging step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - generated_queries: List of SQL queries to merge
                - primary_query: Main SQL query
                - alternative_queries: Optional alternative queries
            config: Configuration parameters for this step including:
                - auto_merge: Whether to automatically merge multiple queries
                - merge_strategy: Strategy for merging (union, join, cte, subquery)
                - validate_merged: Whether to validate the merged query
                - max_merged_queries: Maximum number of queries to merge
                
        Returns:
            Dictionary containing merged query and metadata
        """
        try:
            # Extract required parameters
            generated_queries = input_data.get("generated_queries", [])
            primary_query = input_data.get("primary_query", "")
            alternative_queries = input_data.get("alternative_queries", [])
            
            # Extract configuration
            auto_merge = config.get("auto_merge", True)
            merge_strategy = config.get("merge_strategy", "union")
            validate_merged = config.get("validate_merged", True)
            max_merged_queries = config.get("max_merged_queries", 5)
            
            # Collect all queries to consider for merging
            all_queries = []
            if primary_query and primary_query.strip():
                all_queries.append(primary_query)
            
            # Add alternative queries
            for alt_query in alternative_queries:
                if isinstance(alt_query, dict):
                    query_text = alt_query.get("sql_query", "")
                else:
                    query_text = str(alt_query)
                
                if query_text and query_text.strip():
                    all_queries.append(query_text)
            
            # Add any other generated queries
            for query in generated_queries:
                if isinstance(query, str) and query.strip():
                    all_queries.append(query)
                elif isinstance(query, dict) and query.get("sql_query"):
                    all_queries.append(query["sql_query"])
            
            # Remove duplicates while preserving order
            unique_queries = []
            for query in all_queries:
                normalized_query = self._normalize_query(query)
                if not any(self._normalize_query(existing) == normalized_query for existing in unique_queries):
                    unique_queries.append(query)
            
            # Limit number of queries
            if len(unique_queries) > max_merged_queries:
                unique_queries = unique_queries[:max_merged_queries]
                self._log_warning(f"Limited queries to {max_merged_queries} for merging")
            
            # Determine if merging is needed
            if len(unique_queries) <= 1:
                # No merging needed
                final_query = unique_queries[0] if unique_queries else primary_query
                return self._create_success_result({
                    "merged_query": final_query,
                    "merge_performed": False,
                    "merge_strategy": "single_query",
                    "input_queries_count": len(unique_queries),
                    "merge_metadata": {
                        "reason": "Only one unique query available",
                        "auto_merge": auto_merge
                    }
                })
            
            # Check if auto_merge is enabled
            if not auto_merge:
                # Return primary query without merging
                return self._create_success_result({
                    "merged_query": primary_query,
                    "merge_performed": False,
                    "merge_strategy": "no_merge",
                    "input_queries_count": len(unique_queries),
                    "merge_metadata": {
                        "reason": "Auto merge disabled",
                        "available_queries": len(unique_queries)
                    }
                })
            
            # Perform merging
            merge_result = self._merge_queries(unique_queries, merge_strategy)
            
            if not merge_result.get("success", False):
                # Fall back to primary query if merging fails
                return self._create_success_result({
                    "merged_query": primary_query,
                    "merge_performed": False,
                    "merge_strategy": "fallback",
                    "merge_error": merge_result.get("error_message", "Merge failed"),
                    "input_queries_count": len(unique_queries)
                })
            
            merged_query = merge_result["output_data"]["merged_query"]
            
            # Validate merged query if requested
            validation_result = None
            if validate_merged and merged_query:
                validation_result = self._validate_merged_query(merged_query)
            
            return self._create_success_result({
                "merged_query": merged_query,
                "merge_performed": True,
                "merge_strategy": merge_strategy,
                "input_queries_count": len(unique_queries),
                "validation_result": validation_result,
                "merge_metadata": {
                    **merge_result["output_data"].get("merge_metadata", {}),
                    "original_queries": unique_queries
                }
            })
        
        except Exception as e:
            return self._create_error_result(f"Query merging failed: {e}")
    
    def _merge_queries(self, queries: List[str], merge_strategy: str) -> Dict[str, Any]:
        """Merge multiple SQL queries using the specified strategy."""
        try:
            if merge_sql_queries:
                # Use the tool
                queries_json = json.dumps(queries)
                
                result = self._use_agent_tool(
                    "merge_sql_queries",
                    queries=queries_json,
                    merge_strategy=merge_strategy
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "merged_query": tool_result.get("merged_query", ""),
                            "merge_metadata": tool_result.get("merge_metadata", {})
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "Merge tool failed"))
                else:
                    # Fallback mode
                    return self._merge_queries_fallback(queries, merge_strategy)
            else:
                return self._merge_queries_fallback(queries, merge_strategy)
                
        except Exception as e:
            return self._create_error_result(f"Query merging failed: {e}")
    
    def _validate_merged_query(self, merged_query: str) -> Dict[str, Any]:
        """Validate the merged query."""
        try:
            # Basic validation
            sql_lower = merged_query.lower().strip()
            
            # Check for valid SQL structure
            valid_keywords = ["select", "insert", "update", "delete", "with"]
            starts_correctly = any(sql_lower.startswith(kw) for kw in valid_keywords)
            
            # Check for balanced parentheses
            paren_count = merged_query.count('(') - merged_query.count(')')
            balanced_parens = paren_count == 0
            
            # Check for potential merge artifacts
            merge_issues = []
            if "UNION" in merged_query.upper() and "SELECT" not in merged_query.upper():
                merge_issues.append("UNION found without SELECT statements")
            
            if merged_query.count(';') > 1:
                merge_issues.append("Multiple statements detected")
            
            is_valid = starts_correctly and balanced_parens and not merge_issues
            
            return {
                "valid": is_valid,
                "issues": merge_issues if merge_issues else None,
                "structure_analysis": {
                    "starts_correctly": starts_correctly,
                    "balanced_parentheses": balanced_parens,
                    "statement_count": merged_query.count(';'),
                    "contains_union": "UNION" in merged_query.upper(),
                    "contains_join": "JOIN" in merged_query.upper(),
                    "query_length": len(merged_query)
                }
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {e}"
            }
    
    def _merge_queries_fallback(self, queries: List[str], merge_strategy: str) -> Dict[str, Any]:
        """Fallback method for merging queries."""
        try:
            if len(queries) <= 1:
                return self._create_success_result({
                    "merged_query": queries[0] if queries else "",
                    "merge_metadata": {"fallback_mode": True, "strategy": "single_query"}
                })
            
            # Simple merging strategies
            if merge_strategy == "union":
                # Merge using UNION
                select_queries = []
                for query in queries:
                    query_clean = query.strip().rstrip(';')
                    if query_clean.upper().startswith('SELECT'):
                        select_queries.append(f"({query_clean})")
                
                if select_queries:
                    merged_query = " UNION ".join(select_queries) + ";"
                else:
                    merged_query = queries[0]  # Fallback to first query
            
            elif merge_strategy == "cte":
                # Use Common Table Expressions
                cte_parts = []
                for i, query in enumerate(queries):
                    query_clean = query.strip().rstrip(';')
                    if query_clean.upper().startswith('SELECT'):
                        cte_parts.append(f"cte_{i+1} AS ({query_clean})")
                
                if cte_parts:
                    cte_clause = "WITH " + ", ".join(cte_parts)
                    merged_query = f"{cte_clause} SELECT * FROM cte_1;"
                else:
                    merged_query = queries[0]
            
            else:
                # Default: return first query with comment about others
                merged_query = queries[0]
                if len(queries) > 1:
                    comment = f"-- Additional queries available ({len(queries)-1} alternatives)\\n"
                    merged_query = comment + merged_query
            
            return self._create_success_result({
                "merged_query": merged_query,
                "merge_metadata": {
                    "fallback_mode": True,
                    "strategy_used": merge_strategy,
                    "queries_merged": len(queries)
                }
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback query merging failed: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize a query for comparison."""
        try:
            # Remove comments, extra whitespace, and semicolons for comparison
            normalized = query.strip()
            
            # Remove SQL comments
            lines = normalized.split('\\n')
            cleaned_lines = []
            for line in lines:
                if not line.strip().startswith('--'):
                    cleaned_lines.append(line)
            normalized = '\\n'.join(cleaned_lines)
            
            # Remove extra whitespace and semicolons
            normalized = ' '.join(normalized.split())
            normalized = normalized.rstrip(';')
            
            return normalized.upper()
        except:
            return query.upper()
    
    def _analyze_query_compatibility(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze if queries can be merged together."""
        try:
            analysis = {
                "all_select": True,
                "similar_structure": True,
                "compatible_for_union": True,
                "query_types": []
            }
            
            for query in queries:
                query_upper = query.upper().strip()
                
                if query_upper.startswith('SELECT'):
                    analysis["query_types"].append("SELECT")
                elif query_upper.startswith('INSERT'):
                    analysis["query_types"].append("INSERT")
                    analysis["all_select"] = False
                    analysis["compatible_for_union"] = False
                elif query_upper.startswith('UPDATE'):
                    analysis["query_types"].append("UPDATE")
                    analysis["all_select"] = False
                    analysis["compatible_for_union"] = False
                elif query_upper.startswith('DELETE'):
                    analysis["query_types"].append("DELETE")
                    analysis["all_select"] = False
                    analysis["compatible_for_union"] = False
                else:
                    analysis["query_types"].append("OTHER")
                    analysis["all_select"] = False
                    analysis["compatible_for_union"] = False
            
            # Check structure similarity for SELECT queries
            if analysis["all_select"]:
                # Simple check: count SELECT clauses
                select_counts = [query.upper().count('SELECT') for query in queries]
                analysis["similar_structure"] = len(set(select_counts)) == 1
            
            return analysis
            
        except Exception:
            return {
                "all_select": False,
                "similar_structure": False,
                "compatible_for_union": False,
                "query_types": ["UNKNOWN"] * len(queries)
            }