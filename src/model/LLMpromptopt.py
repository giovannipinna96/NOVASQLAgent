#!/usr/bin/env python3
"""
LLM Prompt Optimizer

A module for optimizing and enhancing prompts to improve LLM performance.
Uses transformer models to analyze and refine input prompts for better results.
"""

import os
import re
import json
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    Pipeline,
    GenerationConfig
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class OptimizationStrategy(Enum):
    """Different prompt optimization strategies."""
    DETAILED_EXPANSION = "detailed_expansion"
    CONTEXT_ENHANCEMENT = "context_enhancement"
    INSTRUCTION_CLARIFICATION = "instruction_clarification"
    EXAMPLE_ENRICHMENT = "example_enrichment"
    STRUCTURED_FORMAT = "structured_format"
    MULTI_STEP_BREAKDOWN = "multi_step_breakdown"


@dataclass
class PromptOptimizationConfig:
    """Configuration for prompt optimization."""
    model_name: str = "microsoft/phi-4-mini-instruct"
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.DETAILED_EXPANSION
    include_reasoning: bool = True
    preserve_intent: bool = True
    add_examples: bool = True
    structured_output: bool = True
    device: Optional[str] = None
    torch_dtype: torch.dtype = torch.float16


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""
    original_prompt: str
    optimized_prompt: str
    optimization_strategy: OptimizationStrategy
    reasoning: Optional[str] = None
    confidence_score: float = 0.0
    improvements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptOptimizer:
    """Main class for optimizing prompts using LLMs."""
    
    def __init__(self, config: Optional[PromptOptimizationConfig] = None):
        """Initialize the prompt optimizer.
        
        Args:
            config: Configuration for optimization
        """
        self.config = config or PromptOptimizationConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._setup_device()
        
    def _setup_device(self):
        """Setup device for model loading."""
        if self.config.device is None:
            if torch.cuda.is_available():
                self.config.device = "cuda"
            else:
                self.config.device = "cpu"
        
        logger.info(f"Using device: {self.config.device}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager"
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.config.device == "cuda" else None,
                torch_dtype=self.config.torch_dtype
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_optimization_prompt_template(self, strategy: OptimizationStrategy) -> str:
        """Get the prompt template for specific optimization strategy."""
        templates = {
            OptimizationStrategy.DETAILED_EXPANSION: """
You are an expert prompt engineer. Your task is to optimize the given prompt by making it more detailed, specific, and comprehensive while preserving the original intent.

Original prompt: "{original_prompt}"

Please optimize this prompt by:
1. Adding specific details and context
2. Clarifying ambiguous terms
3. Including relevant background information
4. Specifying desired output format
5. Adding helpful constraints or guidelines

Provide the optimized prompt in a clear, structured format that will produce better results from an LLM.

Optimized Prompt:""",

            OptimizationStrategy.CONTEXT_ENHANCEMENT: """
You are a prompt optimization expert. Enhance the given prompt by adding relevant context and background information.

Original prompt: "{original_prompt}"

Optimize this prompt by:
1. Adding relevant context and background
2. Explaining key concepts or terminology
3. Providing necessary assumptions
4. Including domain-specific information
5. Setting appropriate expectations

Optimized Prompt:""",

            OptimizationStrategy.INSTRUCTION_CLARIFICATION: """
You are a prompt clarity specialist. Improve the given prompt by making instructions clearer and more precise.

Original prompt: "{original_prompt}"

Enhance clarity by:
1. Using specific, unambiguous language
2. Breaking down complex requests into clear steps
3. Defining technical terms
4. Providing explicit instructions
5. Removing potential misinterpretations

Optimized Prompt:""",

            OptimizationStrategy.EXAMPLE_ENRICHMENT: """
You are a prompt enhancement expert. Improve the given prompt by adding relevant examples and demonstrations.

Original prompt: "{original_prompt}"

Enhance with examples by:
1. Adding relevant use case examples
2. Providing sample inputs and expected outputs
3. Including edge case scenarios
4. Demonstrating proper format
5. Showing good and bad examples where applicable

Optimized Prompt:""",

            OptimizationStrategy.STRUCTURED_FORMAT: """
You are a prompt structuring expert. Reorganize and optimize the given prompt with clear structure and formatting.

Original prompt: "{original_prompt}"

Create a well-structured prompt with:
1. Clear sections and headers
2. Logical flow and organization
3. Proper formatting (bullets, numbers, etc.)
4. Distinct input/output specifications
5. Easy-to-follow structure

Optimized Prompt:""",

            OptimizationStrategy.MULTI_STEP_BREAKDOWN: """
You are a prompt decomposition expert. Break down the given prompt into clear, logical steps.

Original prompt: "{original_prompt}"

Optimize by creating a multi-step approach:
1. Identify the main components of the task
2. Break into logical, sequential steps
3. Define clear objectives for each step
4. Specify expected outcomes
5. Create a structured workflow

Optimized Prompt:"""
        }
        
        return templates.get(strategy, templates[OptimizationStrategy.DETAILED_EXPANSION])
    
    def _generate_optimization_prompt(self, original_prompt: str, strategy: OptimizationStrategy) -> str:
        """Generate the optimization prompt."""
        template = self._get_optimization_prompt_template(strategy)
        return template.format(original_prompt=original_prompt)
    
    def _extract_optimized_prompt(self, generated_text: str) -> str:
        """Extract the optimized prompt from generated text."""
        # Look for "Optimized Prompt:" marker
        if "Optimized Prompt:" in generated_text:
            parts = generated_text.split("Optimized Prompt:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        # Fallback: return the last substantial paragraph
        lines = generated_text.strip().split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if non_empty_lines:
            # Try to find a section that looks like an optimized prompt
            for i, line in enumerate(non_empty_lines):
                if len(line) > 50 and not line.endswith(':'):
                    return '\n'.join(non_empty_lines[i:])
        
        return generated_text.strip()
    
    def _calculate_confidence_score(self, original_prompt: str, optimized_prompt: str) -> float:
        """Calculate confidence score for optimization."""
        # Simple heuristic based on length increase and structure
        length_ratio = len(optimized_prompt) / max(len(original_prompt), 1)
        
        # Check for structural improvements
        structure_score = 0
        if ':' in optimized_prompt: structure_score += 0.2
        if '\n' in optimized_prompt and '\n' not in original_prompt: structure_score += 0.2
        if any(word in optimized_prompt.lower() for word in ['specific', 'detailed', 'example', 'format']): structure_score += 0.2
        
        # Combine factors
        confidence = min(0.5 + (length_ratio - 1) * 0.3 + structure_score, 1.0)
        return max(confidence, 0.1)  # Minimum confidence of 0.1
    
    def _identify_improvements(self, original_prompt: str, optimized_prompt: str) -> List[str]:
        """Identify specific improvements made to the prompt."""
        improvements = []
        
        # Length-based improvements
        if len(optimized_prompt) > len(original_prompt) * 1.5:
            improvements.append("Significantly expanded content and detail")
        elif len(optimized_prompt) > len(original_prompt):
            improvements.append("Added additional detail and context")
        
        # Structure improvements
        if ':' in optimized_prompt and ':' not in original_prompt:
            improvements.append("Added structured formatting")
        
        if optimized_prompt.count('\n') > original_prompt.count('\n'):
            improvements.append("Improved organization and readability")
        
        # Content improvements
        improvement_keywords = {
            'example': "Added examples for clarity",
            'specific': "Made instructions more specific",
            'format': "Specified output format",
            'step': "Broke down into clear steps",
            'context': "Added relevant context"
        }
        
        optimized_lower = optimized_prompt.lower()
        for keyword, improvement in improvement_keywords.items():
            if keyword in optimized_lower and keyword not in original_prompt.lower():
                improvements.append(improvement)
        
        return improvements if improvements else ["General optimization and enhancement"]
    
    def optimize_prompt(
        self, 
        prompt: str, 
        strategy: Optional[OptimizationStrategy] = None
    ) -> OptimizationResult:
        """Optimize a single prompt.
        
        Args:
            prompt: The original prompt to optimize
            strategy: Optimization strategy to use
            
        Returns:
            OptimizationResult containing the optimized prompt and metadata
        """
        if self.pipeline is None:
            self.load_model()
        
        strategy = strategy or self.config.optimization_strategy
        
        logger.info(f"Optimizing prompt with strategy: {strategy.value}")
        
        try:
            # Generate optimization prompt
            optimization_prompt = self._generate_optimization_prompt(prompt, strategy)
            
            # Generate optimized prompt
            generation_config = GenerationConfig(
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            outputs = self.pipeline(
                optimization_prompt,
                generation_config=generation_config,
                return_full_text=False,
                num_return_sequences=1
            )
            
            generated_text = outputs[0]['generated_text']
            optimized_prompt = self._extract_optimized_prompt(generated_text)
            
            # Calculate confidence and improvements
            confidence_score = self._calculate_confidence_score(prompt, optimized_prompt)
            improvements = self._identify_improvements(prompt, optimized_prompt)
            
            result = OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                optimization_strategy=strategy,
                reasoning=generated_text if self.config.include_reasoning else None,
                confidence_score=confidence_score,
                improvements=improvements,
                metadata={
                    'model_name': self.config.model_name,
                    'temperature': self.config.temperature,
                    'original_length': len(prompt),
                    'optimized_length': len(optimized_prompt),
                    'length_ratio': len(optimized_prompt) / len(prompt)
                }
            )
            
            logger.info(f"Optimization completed. Confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            # Return original prompt as fallback
            return OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=prompt,
                optimization_strategy=strategy,
                confidence_score=0.0,
                improvements=[],
                metadata={'error': str(e)}
            )
    
    def optimize_prompts(
        self, 
        prompts: List[str], 
        strategy: Optional[OptimizationStrategy] = None
    ) -> List[OptimizationResult]:
        """Optimize multiple prompts.
        
        Args:
            prompts: List of prompts to optimize
            strategy: Optimization strategy to use
            
        Returns:
            List of OptimizationResult objects
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Optimizing prompt {i+1}/{len(prompts)}")
            result = self.optimize_prompt(prompt, strategy)
            results.append(result)
        
        return results
    
    def compare_prompts(self, original: str, optimized: str) -> Dict[str, Any]:
        """Compare original and optimized prompts.
        
        Args:
            original: Original prompt
            optimized: Optimized prompt
            
        Returns:
            Comparison metrics and analysis
        """
        return {
            'length_increase': len(optimized) - len(original),
            'length_ratio': len(optimized) / len(original) if len(original) > 0 else 0,
            'word_count_increase': len(optimized.split()) - len(original.split()),
            'structure_added': optimized.count('\n') - original.count('\n'),
            'specificity_indicators': sum(1 for word in ['specific', 'detailed', 'example', 'format', 'step'] 
                                        if word in optimized.lower() and word not in original.lower()),
            'original_preview': original[:100] + "..." if len(original) > 100 else original,
            'optimized_preview': optimized[:100] + "..." if len(optimized) > 100 else optimized
        }
    
    def save_results(self, results: List[OptimizationResult], filepath: str):
        """Save optimization results to file.
        
        Args:
            results: List of optimization results
            filepath: Path to save results
        """
        try:
            data = []
            for result in results:
                data.append({
                    'original_prompt': result.original_prompt,
                    'optimized_prompt': result.optimized_prompt,
                    'strategy': result.optimization_strategy.value,
                    'confidence_score': result.confidence_score,
                    'improvements': result.improvements,
                    'metadata': result.metadata
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> List[OptimizationResult]:
        """Load optimization results from file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            List of OptimizationResult objects
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = []
            for item in data:
                result = OptimizationResult(
                    original_prompt=item['original_prompt'],
                    optimized_prompt=item['optimized_prompt'],
                    optimization_strategy=OptimizationStrategy(item['strategy']),
                    confidence_score=item['confidence_score'],
                    improvements=item['improvements'],
                    metadata=item['metadata']
                )
                results.append(result)
            
            logger.info(f"Loaded {len(results)} results from {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return []


def create_optimizer(
    model_name: str = "microsoft/phi-4-mini-instruct",
    strategy: OptimizationStrategy = OptimizationStrategy.DETAILED_EXPANSION,
    **kwargs
) -> PromptOptimizer:
    """Create a prompt optimizer with specified configuration.
    
    Args:
        model_name: Name of the model to use
        strategy: Optimization strategy
        **kwargs: Additional configuration options
        
    Returns:
        Configured PromptOptimizer instance
    """
    config = PromptOptimizationConfig(
        model_name=model_name,
        optimization_strategy=strategy,
        **kwargs
    )
    return PromptOptimizer(config)


def quick_optimize(
    prompt: str, 
    model_name: str = "microsoft/phi-4-mini-instruct",
    strategy: OptimizationStrategy = OptimizationStrategy.DETAILED_EXPANSION
) -> str:
    """Quick prompt optimization function.
    
    Args:
        prompt: Prompt to optimize
        model_name: Model to use for optimization
        strategy: Optimization strategy
        
    Returns:
        Optimized prompt string
    """
    optimizer = create_optimizer(model_name, strategy)
    result = optimizer.optimize_prompt(prompt)
    return result.optimized_prompt


def main():
    """Demo and testing function."""
    print("🚀 LLM Prompt Optimizer Demo")
    print("=" * 50)
    
    # Example prompts to optimize
    test_prompts = [
        "Write a function to sort data",
        "Explain AI",
        "Create a marketing plan",
        "Generate SQL query for user data",
        "Help me debug this code"
    ]
    
    # Test different strategies
    strategies = [
        OptimizationStrategy.DETAILED_EXPANSION,
        OptimizationStrategy.CONTEXT_ENHANCEMENT,
        OptimizationStrategy.INSTRUCTION_CLARIFICATION
    ]
    
    for strategy in strategies:
        print(f"\n🎯 Testing strategy: {strategy.value}")
        print("-" * 30)
        
        optimizer = create_optimizer(strategy=strategy)
        
        # Optimize first test prompt
        result = optimizer.optimize_prompt(test_prompts[0])
        
        print(f"Original: {result.original_prompt}")
        print(f"Optimized: {result.optimized_prompt[:200]}...")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Improvements: {', '.join(result.improvements)}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()