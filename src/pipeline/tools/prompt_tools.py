"""Prompt processing tools for the SQL generation pipeline.

This module provides tools for generating prompt variants and evaluating
their relevance against external descriptions using LLM capabilities.
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
    from model.LLMasJudge import LLMasJudgeSystem, LLMJudge, LLMRewriter
except ImportError:
    # Fallback for when module is not available
    LLMasJudgeSystem = None
    LLMJudge = None
    LLMRewriter = None


@tool
def generate_prompt_variants(original_prompt: str, num_variants: int = 3, max_length: int = 500) -> str:
    """
    Generate variants of the original prompt using LLM rewriter.
    
    Args:
        original_prompt: The original prompt to generate variants for
        num_variants: Number of variants to generate (default: 3)
        max_length: Maximum length for each variant (default: 500)
        
    Returns:
        JSON string containing the generated prompt variants
    """
    try:
        if not original_prompt.strip():
            return json.dumps({
                "error": "Original prompt cannot be empty",
                "variants": []
            })
        
        if LLMasJudgeSystem is None:
            return json.dumps({
                "error": "LLMasJudgeSystem not available - check model imports",
                "variants": []
            })
        
        # Initialize the LLM system
        llm_system = LLMasJudgeSystem()
        
        # Generate variants using the existing rewriter
        variants = []
        generation_metadata = {
            "original_length": len(original_prompt),
            "target_variants": num_variants,
            "max_length": max_length,
            "generation_time": datetime.now().isoformat()
        }
        
        try:
            # Use the existing rewrite_text method
            rewriter_result = llm_system.rewrite_text(original_prompt)
            
            if isinstance(rewriter_result, dict) and rewriter_result.get("success", False):
                # Extract the variants from the structured response
                # The existing LLMRewriter returns {original, text1, text2, text3}
                variant_keys = ["text1", "text2", "text3"]
                
                for i, key in enumerate(variant_keys[:num_variants]):
                    if key in rewriter_result:
                        variant = rewriter_result[key]
                        
                        # Ensure variant doesn't exceed max length
                        if len(variant) > max_length:
                            variant = variant[:max_length-3] + "..."
                        
                        variants.append({
                            "variant_id": i + 1,
                            "text": variant,
                            "length": len(variant),
                            "similarity_to_original": _calculate_similarity(original_prompt, variant)
                        })
            
            else:
                # If rewriter returns different format, handle gracefully
                return json.dumps({
                    "error": "Unexpected rewriter response format",
                    "raw_response": str(rewriter_result),
                    "variants": []
                })
        
        except Exception as rewriter_error:
            return json.dumps({
                "error": f"Rewriter execution failed: {rewriter_error}",
                "variants": []
            })
        
        return json.dumps({
            "success": True,
            "original_prompt": original_prompt,
            "variants": variants,
            "metadata": generation_metadata,
            "total_generated": len(variants)
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to generate prompt variants: {e}",
            "variants": []
        })


@tool
def evaluate_prompt_relevance(prompt_variant: str, description: str, confidence_threshold: float = 0.7) -> str:
    """
    Evaluate if a prompt variant is relevant to a given description using LLM judge.
    
    Args:
        prompt_variant: The prompt variant to evaluate
        description: External description to compare against
        confidence_threshold: Minimum confidence for positive relevance (default: 0.7)
        
    Returns:
        JSON string containing relevance evaluation result
    """
    try:
        if not prompt_variant.strip() or not description.strip():
            return json.dumps({
                "error": "Both prompt variant and description must be non-empty",
                "relevant": False
            })
        
        if LLMasJudgeSystem is None:
            return json.dumps({
                "error": "LLMasJudgeSystem not available - check model imports",
                "relevant": False
            })
        
        # Initialize the LLM system
        llm_system = LLMasJudgeSystem()
        
        try:
            # Use the existing judge_relevance method
            judge_result = llm_system.judge_relevance(prompt_variant, description)
            
            # Parse the result based on the existing implementation
            is_relevant = False
            confidence_score = 0.0
            
            if isinstance(judge_result, str):
                # The existing judge returns "yes", "no", or "uncertain"
                is_relevant = judge_result.lower().strip() == "yes"
                confidence_score = 0.9 if judge_result.lower().strip() in ["yes", "no"] else 0.1
            else:
                # If unexpected format, handle gracefully
                is_relevant = False
                confidence_score = 0.0
            
            # Apply confidence threshold
            final_relevant = is_relevant and confidence_score >= confidence_threshold
            
            return json.dumps({
                "success": True,
                "prompt_variant": prompt_variant,
                "description": description,
                "relevant": final_relevant,
                "raw_decision": is_relevant,
                "confidence_score": confidence_score,
                "confidence_threshold": confidence_threshold,
                "evaluation_metadata": {
                    "prompt_length": len(prompt_variant),
                    "description_length": len(description),
                    "evaluation_time": datetime.now().isoformat()
                }
            })
            
        except Exception as judge_error:
            return json.dumps({
                "error": f"Judge execution failed: {judge_error}",
                "relevant": False
            })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to evaluate prompt relevance: {e}",
            "relevant": False
        })


@tool
def get_prompt_metadata(prompt: str) -> str:
    """
    Get metadata and analysis for a given prompt.
    
    Args:
        prompt: The prompt to analyze
        
    Returns:
        JSON string containing prompt metadata
    """
    try:
        if not prompt.strip():
            return json.dumps({
                "error": "Prompt cannot be empty"
            })
        
        # Basic text analysis
        words = prompt.split()
        sentences = prompt.split('.')
        
        # Identify potential SQL keywords
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
            'TABLE', 'DATABASE', 'INDEX', 'VIEW'
        ]
        
        found_sql_keywords = [kw for kw in sql_keywords if kw.lower() in prompt.lower()]
        
        # Identify potential database objects
        db_indicators = ['table', 'column', 'row', 'database', 'schema', 'index']
        found_db_indicators = [ind for ind in db_indicators if ind.lower() in prompt.lower()]
        
        # Complexity estimation
        complexity_score = 0
        if len(words) > 20:
            complexity_score += 1
        if len(found_sql_keywords) > 3:
            complexity_score += 2
        if any(word in prompt.lower() for word in ['complex', 'advanced', 'multiple', 'join']):
            complexity_score += 1
        
        complexity_level = "low" if complexity_score <= 1 else "medium" if complexity_score <= 3 else "high"
        
        return json.dumps({
            "success": True,
            "prompt": prompt,
            "metadata": {
                "character_count": len(prompt),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "sql_keywords_found": found_sql_keywords,
                "db_indicators_found": found_db_indicators,
                "complexity_level": complexity_level,
                "complexity_score": complexity_score,
                "analysis_time": datetime.now().isoformat()
            },
            "suggestions": _get_prompt_suggestions(prompt, found_sql_keywords, complexity_level)
        })
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to analyze prompt metadata: {e}"
        })


def _calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts."""
    try:
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    except:
        return 0.0


def _get_prompt_suggestions(prompt: str, sql_keywords: List[str], complexity: str) -> List[str]:
    """Get suggestions for improving the prompt."""
    suggestions = []
    
    if len(prompt.split()) < 5:
        suggestions.append("Consider adding more detail to the prompt for better SQL generation")
    
    if not sql_keywords:
        suggestions.append("Consider adding SQL-related keywords to clarify the database operation needed")
    
    if complexity == "low":
        suggestions.append("Prompt appears simple - consider adding specificity about tables, columns, or conditions")
    
    if "database" not in prompt.lower() and "table" not in prompt.lower():
        suggestions.append("Consider specifying which database or table the query should target")
    
    if len(prompt) > 300:
        suggestions.append("Long prompt detected - consider breaking into smaller, focused requests")
    
    return suggestions