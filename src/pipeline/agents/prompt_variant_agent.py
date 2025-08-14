"""Prompt variant generation agent for the SQL generation pipeline.

This agent handles generating multiple variants of the original prompt
using LLM rewriter capabilities.
"""

import json
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_agent import BasePipelineAgent

try:
    from model.LLMasJudge import LLMasJudgeSystem
    from ..tools.prompt_tools import (
        generate_prompt_variants,
        get_prompt_metadata
    )
except ImportError:
    # Fallback imports
    LLMasJudgeSystem = None
    generate_prompt_variants = None
    get_prompt_metadata = None


class PromptVariantAgent(BasePipelineAgent):
    """
    Agent responsible for generating prompt variants in the pipeline.
    
    This agent handles:
    - Generating multiple variants of the original prompt
    - Analyzing prompt characteristics
    - Providing metadata about generated variants
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct"):
        super().__init__(model_id=model_id, agent_name="prompt_variant_agent")
    
    def _get_agent_tools(self) -> List:
        """Get prompt-related tools for the agent."""
        tools = []
        
        if generate_prompt_variants:
            tools.append(generate_prompt_variants)
        if get_prompt_metadata:
            tools.append(get_prompt_metadata)
        
        return tools
    
    def execute_step(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prompt variant generation step for the pipeline.
        
        Args:
            input_data: Dictionary containing:
                - original_prompt: The original prompt to generate variants for
                - external_descriptions: Optional list of external descriptions for context
            config: Configuration parameters for this step including:
                - num_variants: Number of variants to generate
                - max_variant_length: Maximum length for each variant
                - similarity_threshold: Minimum similarity threshold
                
        Returns:
            Dictionary containing generated prompt variants and metadata
        """
        try:
            # Extract required parameters
            original_prompt = input_data.get("original_prompt", "")
            external_descriptions = input_data.get("external_descriptions", [])
            
            # Extract configuration
            num_variants = config.get("num_variants", 3)
            max_variant_length = config.get("max_variant_length", 500)
            similarity_threshold = config.get("similarity_threshold", 0.8)
            
            # Validate inputs
            if not original_prompt.strip():
                return self._create_error_result("Original prompt cannot be empty")
            
            # Generate prompt metadata first
            metadata_result = self._analyze_prompt(original_prompt)
            
            # Generate prompt variants
            variants_result = self._generate_variants(
                original_prompt, 
                num_variants, 
                max_variant_length
            )
            
            if not variants_result.get("success", False):
                return variants_result
            
            # Extract variants and enhance with additional analysis
            variants = variants_result["output_data"]["variants"]
            enhanced_variants = self._enhance_variants_analysis(
                original_prompt, 
                variants, 
                similarity_threshold
            )
            
            return self._create_success_result({
                "original_prompt": original_prompt,
                "variants": enhanced_variants,
                "prompt_metadata": metadata_result.get("output_data", {}),
                "generation_stats": {
                    "requested_variants": num_variants,
                    "generated_variants": len(enhanced_variants),
                    "avg_variant_length": sum(len(v["text"]) for v in enhanced_variants) / len(enhanced_variants) if enhanced_variants else 0,
                    "similarity_threshold": similarity_threshold
                }
            })
        
        except Exception as e:
            return self._create_error_result(f"Prompt variant generation failed: {e}")
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the original prompt to get metadata."""
        try:
            if get_prompt_metadata:
                # Use the tool
                result = self._use_agent_tool("get_prompt_metadata", prompt=prompt)
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result(tool_result["metadata"])
                    else:
                        return self._create_error_result(tool_result.get("error", "Metadata analysis failed"))
                else:
                    # Fallback mode
                    return self._analyze_prompt_fallback(prompt)
            else:
                return self._analyze_prompt_fallback(prompt)
                
        except Exception as e:
            return self._create_error_result(f"Prompt analysis failed: {e}")
    
    def _generate_variants(self, original_prompt: str, num_variants: int, max_length: int) -> Dict[str, Any]:
        """Generate variants of the original prompt."""
        try:
            if generate_prompt_variants:
                # Use the tool
                result = self._use_agent_tool(
                    "generate_prompt_variants",
                    original_prompt=original_prompt,
                    num_variants=num_variants,
                    max_length=max_length
                )
                
                if result.get("success", False):
                    tool_result = json.loads(result["result"]) if isinstance(result["result"], str) else result["result"]
                    
                    if tool_result.get("success", False):
                        return self._create_success_result({
                            "variants": tool_result["variants"],
                            "total_generated": tool_result.get("total_generated", 0),
                            "generation_metadata": tool_result.get("metadata", {})
                        })
                    else:
                        return self._create_error_result(tool_result.get("error", "Variant generation failed"))
                else:
                    # Fallback mode
                    return self._generate_variants_fallback(original_prompt, num_variants, max_length)
            else:
                return self._generate_variants_fallback(original_prompt, num_variants, max_length)
                
        except Exception as e:
            return self._create_error_result(f"Variant generation failed: {e}")
    
    def _enhance_variants_analysis(self, original_prompt: str, variants: List[Dict], 
                                 similarity_threshold: float) -> List[Dict[str, Any]]:
        """Enhance variants with additional analysis."""
        enhanced_variants = []
        
        for i, variant in enumerate(variants):
            try:
                variant_text = variant.get("text", "")
                
                # Calculate additional metrics
                word_overlap = self._calculate_word_overlap(original_prompt, variant_text)
                length_ratio = len(variant_text) / len(original_prompt) if original_prompt else 0
                complexity_score = self._estimate_complexity(variant_text)
                
                # Determine quality assessment
                quality_score = (
                    variant.get("similarity_to_original", 0) * 0.4 +
                    word_overlap * 0.3 +
                    min(length_ratio, 1.0) * 0.2 +
                    min(complexity_score / 5.0, 1.0) * 0.1
                )
                
                enhanced_variant = {
                    "variant_id": variant.get("variant_id", i + 1),
                    "text": variant_text,
                    "length": variant.get("length", len(variant_text)),
                    "similarity_to_original": variant.get("similarity_to_original", 0),
                    "word_overlap": word_overlap,
                    "length_ratio": length_ratio,
                    "complexity_score": complexity_score,
                    "quality_score": quality_score,
                    "meets_threshold": variant.get("similarity_to_original", 0) >= similarity_threshold,
                    "analysis": {
                        "quality_assessment": self._assess_quality(quality_score),
                        "length_assessment": self._assess_length(length_ratio),
                        "similarity_assessment": self._assess_similarity(variant.get("similarity_to_original", 0))
                    }
                }
                
                enhanced_variants.append(enhanced_variant)
                
            except Exception as e:
                # If enhancement fails, keep the original variant
                self._log_warning(f"Failed to enhance variant {i}: {e}")
                enhanced_variants.append(variant)
        
        return enhanced_variants
    
    def _analyze_prompt_fallback(self, prompt: str) -> Dict[str, Any]:
        """Fallback method for prompt analysis."""
        try:
            words = prompt.split()
            sentences = prompt.split('.')
            
            # Basic analysis
            analysis = {
                "character_count": len(prompt),
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "complexity_level": "medium",
                "fallback_mode": True
            }
            
            return self._create_success_result(analysis)
            
        except Exception as e:
            return self._create_error_result(f"Fallback prompt analysis failed: {e}")
    
    def _generate_variants_fallback(self, original_prompt: str, num_variants: int, max_length: int) -> Dict[str, Any]:
        """Fallback method for generating variants."""
        try:
            # Simple fallback - create variants by modifying the original prompt
            variants = []
            
            for i in range(num_variants):
                # Simple modifications
                if i == 0:
                    variant_text = f"Please help with: {original_prompt}"
                elif i == 1:
                    variant_text = f"I need assistance with the following: {original_prompt}"
                else:
                    variant_text = f"Could you provide guidance on: {original_prompt}"
                
                # Truncate if too long
                if len(variant_text) > max_length:
                    variant_text = variant_text[:max_length-3] + "..."
                
                variants.append({
                    "variant_id": i + 1,
                    "text": variant_text,
                    "length": len(variant_text),
                    "similarity_to_original": 0.8  # Estimate
                })
            
            return self._create_success_result({
                "variants": variants,
                "total_generated": len(variants),
                "fallback_mode": True
            })
            
        except Exception as e:
            return self._create_error_result(f"Fallback variant generation failed: {e}")
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
        except:
            return 0.0
    
    def _estimate_complexity(self, text: str) -> int:
        """Estimate text complexity on a scale of 1-10."""
        try:
            complexity = 0
            
            # Length factor
            if len(text) > 100:
                complexity += 2
            if len(text) > 200:
                complexity += 1
            
            # Word count factor
            words = text.split()
            if len(words) > 20:
                complexity += 2
            if len(words) > 40:
                complexity += 1
            
            # Technical terms
            technical_terms = ["database", "query", "table", "select", "join", "where"]
            found_terms = sum(1 for term in technical_terms if term.lower() in text.lower())
            complexity += min(found_terms, 3)
            
            return min(complexity, 10)
        except:
            return 5  # Default medium complexity
    
    def _assess_quality(self, quality_score: float) -> str:
        """Assess variant quality based on score."""
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_length(self, length_ratio: float) -> str:
        """Assess variant length compared to original."""
        if length_ratio < 0.5:
            return "much_shorter"
        elif length_ratio < 0.8:
            return "shorter"
        elif length_ratio <= 1.2:
            return "similar"
        elif length_ratio <= 1.5:
            return "longer"
        else:
            return "much_longer"
    
    def _assess_similarity(self, similarity_score: float) -> str:
        """Assess similarity level."""
        if similarity_score >= 0.9:
            return "very_high"
        elif similarity_score >= 0.7:
            return "high"
        elif similarity_score >= 0.5:
            return "medium"
        elif similarity_score >= 0.3:
            return "low"
        else:
            return "very_low"