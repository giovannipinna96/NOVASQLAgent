"""Relevance evaluation agent for the SQL generation pipeline.

This agent evaluates the relevance of prompt variants against external descriptions
using LLM judge capabilities to determine which descriptions are relevant.
"""

import json
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent
from model.LLMasJudge import LLMasJudgeSystem

try:
    from ..tools.prompt_tools import evaluate_prompt_relevance
except ImportError:
    # Fallback imports
    evaluate_prompt_relevance = None


class RelevanceEvaluationAgent(BasePipelineAgent):
    """
    Agent responsible for evaluating relevance in the pipeline.
    
    This agent handles:
    - Evaluating prompt variants against external descriptions
    - Determining which descriptions are relevant for SQL generation
    - Providing confidence scores and evaluation metadata
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="relevance_evaluation_agent")
    
    def _get_agent_tools(self) -> List:
        """Get relevance evaluation tools for the agent."""
        tools = []
        
        if evaluate_prompt_relevance:
            tools.append(evaluate_prompt_relevance)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute relevance evaluation step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - prompt_variants: List of prompt variants to evaluate
                - external_descriptions: List of external descriptions to check against
                - original_prompt: Original prompt for reference
            config: Configuration parameters for this step including:
                - confidence_threshold: Minimum confidence for positive relevance
                - require_unanimous: Whether all variants must agree for relevance
                - max_descriptions: Maximum number of descriptions to process
                
        Returns:
            Dictionary containing relevance evaluation results
        """
        try:
            # Extract required parameters
            prompt_variants = input_data.get("prompt_variants", [])
            external_descriptions = input_data.get("external_descriptions", [])
            original_prompt = input_data.get("original_prompt", "")
            
            # Extract configuration
            confidence_threshold = config.get("confidence_threshold", 0.7)
            require_unanimous = config.get("require_unanimous", False)
            max_descriptions = config.get("max_descriptions", 10)
            
            # Validate inputs
            if not prompt_variants:
                return self._create_error_result("No prompt variants provided for evaluation")
            
            if not external_descriptions:
                return self._create_error_result("No external descriptions provided for evaluation")
            
            # Limit descriptions if needed
            if len(external_descriptions) > max_descriptions:
                external_descriptions = external_descriptions[:max_descriptions]
                self._log_warning(f"Limited external descriptions to {max_descriptions}")
            
            # Perform relevance evaluation
            evaluation_matrix, evaluation_details = self._evaluate_relevance_matrix(
                prompt_variants, external_descriptions, confidence_threshold
            )
            
            # Determine relevant descriptions
            relevant_descriptions = self._determine_relevant_descriptions(
                evaluation_matrix, external_descriptions, require_unanimous
            )
            
            # Calculate evaluation statistics
            stats = self._calculate_evaluation_stats(
                evaluation_matrix, evaluation_details, relevant_descriptions
            )
            
            return self._create_success_result({
                "relevant_descriptions": relevant_descriptions,
                "evaluation_matrix": evaluation_matrix,
                "evaluation_details": evaluation_details,
                "evaluation_stats": stats,
                "config_used": {
                    "confidence_threshold": confidence_threshold,
                    "require_unanimous": require_unanimous,
                    "max_descriptions": max_descriptions
                }
            })
        
        except Exception as e:
            return self._create_error_result(f"Relevance evaluation failed: {e}")
    
    def _evaluate_relevance_matrix(self, prompt_variants: List[Dict], external_descriptions: List[str], 
                                 confidence_threshold: float) -> Tuple[List[List[str]], List[Dict]]:
        """
        Evaluate relevance between all variants and descriptions.
        
        Returns:
            Tuple of (evaluation_matrix, evaluation_details)
        """
        evaluation_matrix = []
        evaluation_details = []
        
        for variant_idx, variant in enumerate(prompt_variants):
            variant_text = variant.get("text", "")
            variant_evaluations = []
            
            for desc_idx, description in enumerate(external_descriptions):
                # Evaluate relevance
                relevance_result = self._evaluate_single_relevance(
                    variant_text, description, confidence_threshold
                )
                
                # Store matrix result (yes/no)
                is_relevant = relevance_result.get("relevant", False)
                variant_evaluations.append("yes" if is_relevant else "no")
                
                # Store detailed result
                evaluation_details.append({
                    "variant_id": variant.get("variant_id", variant_idx + 1),
                    "variant_text": variant_text,
                    "description_id": desc_idx + 1,
                    "description": description,
                    "relevant": is_relevant,
                    "confidence_score": relevance_result.get("confidence_score", 0.0),
                    "raw_decision": relevance_result.get("raw_decision", False),
                    "evaluation_metadata": relevance_result.get("evaluation_metadata", {})
                })
            
            evaluation_matrix.append(variant_evaluations)
        
        return evaluation_matrix, evaluation_details
    
    def _evaluate_single_relevance(self, variant_text: str, description: str, 
                                 confidence_threshold: float) -> Dict[str, Any]:
        """Evaluate relevance between a single variant and description."""
        try:
            if evaluate_prompt_relevance:
                # Use the tool
                result = self._use_agent_tool(
                    "evaluate_prompt_relevance",
                    prompt_variant=variant_text,
                    description=description,
                    confidence_threshold=confidence_threshold
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return {
                            "relevant": tool_result.get("relevant", False),
                            "confidence_score": tool_result.get("confidence_score", 0.0),
                            "raw_decision": tool_result.get("raw_decision", False),
                            "evaluation_metadata": tool_result.get("evaluation_metadata", {})
                        }
                    else:
                        self._log_warning(f"Tool evaluation failed: {tool_result.get('error', 'Unknown error')}")
                        return self._evaluate_relevance_fallback(variant_text, description)
                else:
                    # Fallback mode
                    return self._evaluate_relevance_fallback(variant_text, description)
            else:
                return self._evaluate_relevance_fallback(variant_text, description)
                
        except Exception as e:
            self._log_error(f"Single relevance evaluation failed: {e}")
            return self._evaluate_relevance_fallback(variant_text, description)
    
    def _determine_relevant_descriptions(self, evaluation_matrix: List[List[str]], 
                                       external_descriptions: List[str], 
                                       require_unanimous: bool) -> List[str]:
        """Determine which descriptions are relevant based on evaluation results."""
        relevant_descriptions = []
        
        # Transpose matrix to get description-wise evaluations
        num_descriptions = len(external_descriptions)
        num_variants = len(evaluation_matrix)
        
        for desc_idx in range(num_descriptions):
            # Get all evaluations for this description
            desc_evaluations = [evaluation_matrix[var_idx][desc_idx] for var_idx in range(num_variants)]
            
            # Count positive evaluations
            positive_count = desc_evaluations.count("yes")
            
            # Determine relevance based on strategy
            if require_unanimous:
                # All variants must agree
                is_relevant = positive_count == num_variants
            else:
                # Majority vote
                is_relevant = positive_count > (num_variants / 2)
            
            if is_relevant:
                relevant_descriptions.append(external_descriptions[desc_idx])
        
        return relevant_descriptions
    
    def _calculate_evaluation_stats(self, evaluation_matrix: List[List[str]], 
                                  evaluation_details: List[Dict], 
                                  relevant_descriptions: List[str]) -> Dict[str, Any]:
        """Calculate statistics about the evaluation process."""
        try:
            total_evaluations = len(evaluation_details)
            positive_evaluations = sum(1 for detail in evaluation_details if detail["relevant"])
            
            # Variant-wise statistics
            variant_stats = {}
            for detail in evaluation_details:
                variant_id = detail["variant_id"]
                if variant_id not in variant_stats:
                    variant_stats[variant_id] = {"positive": 0, "total": 0, "avg_confidence": 0.0}
                
                variant_stats[variant_id]["total"] += 1
                if detail["relevant"]:
                    variant_stats[variant_id]["positive"] += 1
                
                # Update average confidence
                current_avg = variant_stats[variant_id]["avg_confidence"]
                current_total = variant_stats[variant_id]["total"]
                new_confidence = detail["confidence_score"]
                variant_stats[variant_id]["avg_confidence"] = (
                    (current_avg * (current_total - 1) + new_confidence) / current_total
                )
            
            # Calculate variant agreement
            num_variants = len(evaluation_matrix)
            num_descriptions = len(evaluation_matrix[0]) if evaluation_matrix else 0
            
            agreement_count = 0
            for desc_idx in range(num_descriptions):
                desc_evaluations = [evaluation_matrix[var_idx][desc_idx] for var_idx in range(num_variants)]
                if len(set(desc_evaluations)) == 1:  # All agree
                    agreement_count += 1
            
            agreement_rate = agreement_count / num_descriptions if num_descriptions > 0 else 0
            
            return {
                "total_evaluations": total_evaluations,
                "positive_evaluations": positive_evaluations,
                "positive_rate": positive_evaluations / total_evaluations if total_evaluations > 0 else 0,
                "relevant_descriptions_count": len(relevant_descriptions),
                "total_descriptions": num_descriptions,
                "relevance_rate": len(relevant_descriptions) / num_descriptions if num_descriptions > 0 else 0,
                "variant_agreement_rate": agreement_rate,
                "variant_stats": variant_stats,
                "average_confidence": sum(detail["confidence_score"] for detail in evaluation_details) / total_evaluations if total_evaluations > 0 else 0
            }
            
        except Exception as e:
            self._log_error(f"Failed to calculate evaluation stats: {e}")
            return {"error": str(e)}
    
    def _evaluate_relevance_fallback(self, variant_text: str, description: str) -> Dict[str, Any]:
        """Fallback method for relevance evaluation."""
        try:
            # Simple keyword-based relevance check
            variant_words = set(variant_text.lower().split())
            description_words = set(description.lower().split())
            
            # Calculate word overlap
            overlap = variant_words.intersection(description_words)
            overlap_ratio = len(overlap) / len(variant_words.union(description_words)) if variant_words.union(description_words) else 0
            
            # Simple threshold-based decision
            is_relevant = overlap_ratio > 0.3
            confidence = min(overlap_ratio * 2, 1.0)  # Scale to 0-1
            
            return {
                "relevant": is_relevant,
                "confidence_score": confidence,
                "raw_decision": is_relevant,
                "evaluation_metadata": {
                    "word_overlap_ratio": overlap_ratio,
                    "shared_words": len(overlap),
                    "fallback_mode": True
                }
            }
            
        except Exception as e:
            # Last resort fallback
            return {
                "relevant": False,
                "confidence_score": 0.0,
                "raw_decision": False,
                "evaluation_metadata": {"error": str(e), "fallback_mode": True}
            }