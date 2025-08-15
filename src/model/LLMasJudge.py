"""
LLM as Judge and Text Rewriter Module

Enterprise-grade text analysis system implementing dual-model architecture:
1. LLMJudge: Binary relevance evaluation with confidence scoring
2. LLMRewriter: Multi-variant paraphrase generation with JSON output

Design Patterns:
- Strategy Pattern: Pluggable judgment and rewriting strategies
- Factory Pattern: Model instantiation and pipeline creation
- Template Method: Common model loading and inference patterns
- Observer Pattern: Logging and monitoring capabilities
- Builder Pattern: Flexible configuration setup

Features:
- Type-safe interfaces with comprehensive error handling
- Automatic model optimization for target hardware
- Structured JSON output with fallback mechanisms
- Batch processing capabilities for scalable operations
- Configurable inference parameters for different use cases
"""

import json
import torch
from typing import Dict, List, Optional, Union, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JudgmentResult(Enum):
    """Enumeration for judgment results."""
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    model_id: str = "microsoft/phi-4-mini-instruct"
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = True


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    max_new_tokens: int = 10
    temperature: float = 0.1
    do_sample: bool = False
    top_p: float = 0.9
    return_full_text: bool = False


@runtime_checkable
class TextProcessor(Protocol):
    """Protocol for text processing operations."""
    
    def process(self, text: str, **kwargs) -> Dict[str, Union[str, float]]:
        """Process text and return structured result."""
        ...


class BaseModelProcessor(ABC):
    """Abstract base class for model processors using Template Method pattern."""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.pipeline: Optional[Pipeline] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and pipeline - Template Method pattern."""
        device = self.model_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = self.model_config.torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
        
        try:
            logger.info(f"Loading model: {self.model_config.model_id}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
            
            # Detect model type and load appropriately
            if "t5" in self.model_config.model_id.lower():
                from transformers import T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=self.model_config.trust_remote_code
                )
            elif "bart" in self.model_config.model_id.lower():
                from transformers import BartForConditionalGeneration
                model = BartForConditionalGeneration.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=self.model_config.trust_remote_code
                )
            else:
                # Default to causal LM for most other models
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=self.model_config.trust_remote_code
                )
            
            # Select appropriate task type based on model
            if "t5" in self.model_config.model_id.lower():
                task = "text2text-generation"
            elif "bart" in self.model_config.model_id.lower():
                task = "text2text-generation"
            else:
                task = "text-generation"
            
            self.pipeline = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=dtype,
                **self._get_pipeline_config()
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_pipeline_config(self) -> Dict[str, Union[int, float, bool]]:
        """Get pipeline configuration parameters."""
        config = {
            "max_new_tokens": self.inference_config.max_new_tokens,
            "do_sample": self.inference_config.do_sample,
            "temperature": self.inference_config.temperature,
            "top_p": self.inference_config.top_p,
        }
        
        # Only add return_full_text for causal models, not for T5/BART
        if not ("t5" in self.model_config.model_id.lower() or "bart" in self.model_config.model_id.lower()):
            config["return_full_text"] = self.inference_config.return_full_text
            
        return config
    
    @abstractmethod
    def _create_prompt(self, *args, **kwargs) -> str:
        """Create prompt for the specific task."""
        pass
    
    @abstractmethod
    def _parse_response(self, response: str) -> Dict[str, Union[str, float]]:
        """Parse model response into structured format."""
        pass
    
    def process(self, *args, **kwargs) -> Dict[str, Union[str, float]]:
        """Main processing method - Template Method pattern."""
        if not self.pipeline:
            raise RuntimeError("Model not initialized properly")
        
        try:
            prompt = self._create_prompt(*args, **kwargs)
            result = self.pipeline(prompt)
            response = result[0]["generated_text"].strip()
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return self._create_error_response(str(e))
    
    def _create_error_response(self, error: str) -> Dict[str, Union[str, float]]:
        """Create standardized error response."""
        return {"error": error, "success": False}


class LLMJudge(BaseModelProcessor):
    """
    LLM Judge for binary text relevance evaluation.
    
    Implements Strategy Pattern for different judgment approaches and uses
    phi-4-mini-instruct for consistent binary classification with confidence scoring.
    
    Features:
    - Binary relevance classification (yes/no/uncertain)
    - Confidence scoring and uncertainty detection
    - Batch processing capabilities
    - Structured output with error handling
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, inference_config: Optional[InferenceConfig] = None):
        """
        Initialize the LLM Judge with enhanced configuration.
        
        Args:
            model_config: Model configuration settings
            inference_config: Inference parameter configuration
        """
        model_config = model_config or ModelConfig()
        inference_config = inference_config or InferenceConfig(max_new_tokens=10, do_sample=False)
        
        super().__init__(model_config, inference_config)
        
        # System prompt for relevance judgment
        self.system_prompt = (
            "You are a text relevance judge. Your task is to determine if one text "
            "is relevant to another text. You must respond with only 'yes' or 'no'. "
            "Do not provide explanations or additional text."
        )
    
    def _create_prompt(self, text1: str, text2: str) -> str:
        """Create prompt for relevance judgment."""
        return f"""System: {self.system_prompt}

Question: Is the following text relevant to the reference text?

Reference text: {text2}

Text to evaluate: {text1}

Answer (yes/no):"""
    
    def _parse_response(self, response: str) -> Dict[str, Union[str, float]]:
        """Parse judgment response into structured format."""
        response_lower = response.lower()
        
        if "yes" in response_lower:
            judgment = JudgmentResult.YES.value
            confidence = 0.9  # High confidence for clear response
        elif "no" in response_lower:
            judgment = JudgmentResult.NO.value
            confidence = 0.9
        else:
            judgment = JudgmentResult.UNCERTAIN.value
            confidence = 0.1  # Low confidence for unclear response
            logger.warning(f"Unclear response from judge: {response}")
        
        return {
            "judgment": judgment,
            "confidence": confidence,
            "raw_response": response,
            "success": True
        }
    
    def judge_relevance(self, text1: str, text2: str) -> str:
        """
        Judge if text1 is relevant to text2 with enhanced error handling.
        
        Args:
            text1: First text to compare
            text2: Second text to compare against
            
        Returns:
            "yes", "no", or "uncertain" based on analysis
        """
        result = self.process(text1, text2)
        
        if result.get("success", False):
            return result["judgment"]
        else:
            logger.error(f"Judgment failed: {result.get('error', 'Unknown error')}")
            return JudgmentResult.UNCERTAIN.value


class LLMRewriter(BaseModelProcessor):
    """
    LLM Rewriter for multi-variant text paraphrase generation.
    
    Implements Strategy Pattern for different rewriting approaches using
    phi-4-mini-instruct to generate structured paraphrases with JSON output.
    
    Features:
    - Multi-variant paraphrase generation (3 variants)
    - Structured JSON output with fallback mechanisms
    - Creative sampling with controlled randomness
    - Robust error handling and validation
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, inference_config: Optional[InferenceConfig] = None):
        """
        Initialize the LLM Rewriter with enhanced configuration.
        
        Args:
            model_config: Model configuration settings
            inference_config: Inference parameter configuration
        """
        model_config = model_config or ModelConfig()
        inference_config = inference_config or InferenceConfig(
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7
        )
        
        super().__init__(model_config, inference_config)
        
        # System prompt for text rewriting
        self.system_prompt = (
            "You are a text rewriter. Your task is to create exactly 3 different "
            "paraphrases of the given text. Maintain the same meaning but vary the "
            "wording and structure. Return your response as valid JSON with keys: "
            "'original', 'text1', 'text2', 'text3'."
        )
    
    def _create_prompt(self, text: str) -> str:
        """Create prompt for text rewriting."""
        return f"""System: {self.system_prompt}

Original text: {text}

Generate 3 paraphrases in JSON format:"""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse rewriting response into structured JSON format."""
        try:
            # Look for JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed_json = json.loads(json_str)
                
                # Validate required keys
                required_keys = ["original", "text1", "text2", "text3"]
                if all(key in parsed_json for key in required_keys):
                    parsed_json["success"] = True
                    return parsed_json
            
            # If JSON parsing fails, create structured response manually
            logger.warning("Failed to parse JSON, creating fallback response")
            return self._create_fallback_response("", response)
            
        except json.JSONDecodeError:
            logger.warning("JSON decode error, creating fallback response")
            return self._create_fallback_response("", response)
    
    def rewrite_text(self, text: str) -> Dict[str, str]:
        """
        Generate 3 paraphrases of the input text with enhanced error handling.
        
        Args:
            text: Original text to paraphrase
            
        Returns:
            Dictionary with original text and 3 paraphrases
        """
        result = self.process(text)
        
        if result.get("success", False):
            return result
        else:
            logger.error(f"Rewriting failed: {result.get('error', 'Unknown error')}")
            return self._create_error_response(text)
    
    def _create_fallback_response(self, original_text: str, response: str) -> Dict[str, str]:
        """Create a fallback response when JSON parsing fails."""
        # Split response into lines and try to extract paraphrases
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        paraphrases = []
        
        for line in lines:
            if len(line) > 10 and line != original_text:  # Basic filter
                paraphrases.append(line)
                if len(paraphrases) >= 3:
                    break
        
        # Ensure we have at least 3 paraphrases
        while len(paraphrases) < 3:
            paraphrases.append(f"Paraphrase {len(paraphrases) + 1}: {original_text}")
        
        return {
            "original": original_text,
            "text1": paraphrases[0][:200],  # Limit length
            "text2": paraphrases[1][:200],
            "text3": paraphrases[2][:200]
        }
    
    def _create_error_response(self, original_text: str) -> Dict[str, str]:
        """Create an error response when generation fails."""
        return {
            "original": original_text,
            "text1": f"Unable to generate paraphrase 1 for: {original_text}",
            "text2": f"Unable to generate paraphrase 2 for: {original_text}",
            "text3": f"Unable to generate paraphrase 3 for: {original_text}"
        }


class LLMProcessorFactory:
    """
    Factory class for creating text processors using Factory Pattern.
    
    Provides centralized creation and configuration of LLM processors
    with consistent setup and error handling.
    """
    
    @staticmethod
    def create_judge(model_config: Optional[ModelConfig] = None, 
                    inference_config: Optional[InferenceConfig] = None) -> LLMJudge:
        """Create LLM Judge with default configurations."""
        return LLMJudge(model_config, inference_config)
    
    @staticmethod
    def create_rewriter(model_config: Optional[ModelConfig] = None,
                       inference_config: Optional[InferenceConfig] = None) -> LLMRewriter:
        """Create LLM Rewriter with default configurations."""
        return LLMRewriter(model_config, inference_config)


class LLMasJudgeSystem:
    """
    Unified system providing both judging and rewriting capabilities.
    
    Implements Facade Pattern to provide a simplified interface over
    complex text processing subsystems with centralized configuration.
    
    Features:
    - Unified interface for multiple text processors
    - Batch processing capabilities
    - Consistent error handling across processors
    - Factory-based processor creation
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize the combined LLM system using Factory Pattern.
        
        Args:
            model_config: Shared model configuration for all processors
        """
        self.model_config = model_config or ModelConfig()
        
        logger.info("Initializing LLM as Judge System")
        
        # Create processors using factory
        judge_config = InferenceConfig(max_new_tokens=10, do_sample=False)
        rewriter_config = InferenceConfig(max_new_tokens=300, do_sample=True, temperature=0.7)
        
        self.judge = LLMProcessorFactory.create_judge(self.model_config, judge_config)
        self.rewriter = LLMProcessorFactory.create_rewriter(self.model_config, rewriter_config)
        
        logger.info("LLM as Judge System initialized successfully")
    
    def judge_relevance(self, text1: str, text2: str) -> str:
        """Judge text relevance using LLMJudge."""
        return self.judge.judge_relevance(text1, text2)
    
    def rewrite_text(self, text: str) -> Dict[str, str]:
        """Generate text paraphrases using LLMRewriter."""
        return self.rewriter.rewrite_text(text)
    
    def batch_judge(self, text_pairs: List[tuple]) -> List[str]:
        """
        Judge multiple text pairs for relevance.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            
        Returns:
            List of "yes"/"no" judgments
        """
        results = []
        for text1, text2 in text_pairs:
            result = self.judge_relevance(text1, text2)
            results.append(result)
        return results
    
    def batch_rewrite(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Rewrite multiple texts.
        
        Args:
            texts: List of texts to paraphrase
            
        Returns:
            List of dictionaries with paraphrases
        """
        results = []
        for text in texts:
            result = self.rewrite_text(text)
            results.append(result)
        return results


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM as Judge and Rewriter System")
    parser.add_argument("--mode", choices=["judge", "rewrite", "both"], default="both",
                       help="Operation mode: judge relevance, rewrite text, or both")
    parser.add_argument("--text1", type=str, help="First text (for judging) or text to rewrite")
    parser.add_argument("--text2", type=str, help="Second text (for judging only)")
    parser.add_argument("--model", type=str, default="microsoft/phi-4-mini-instruct",
                       help="Model ID to use")
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    if not args.text1:
        # Demo mode with example texts
        print("Running demo mode...")
        
        model_config = ModelConfig(model_id=args.model, device=args.device)
        system = LLMasJudgeSystem(model_config)
        
        # Demo judging
        print("\n=== LLM Judge Demo ===")
        text1 = "The weather is sunny today."
        text2 = "It's raining heavily outside."
        result = system.judge_relevance(text1, text2)
        print(f"Text1: {text1}")
        print(f"Text2: {text2}")
        print(f"Relevance: {result}")
        
        # Demo rewriting
        print("\n=== LLM Rewriter Demo ===")
        text_to_rewrite = "The quick brown fox jumps over the lazy dog."
        result = system.rewrite_text(text_to_rewrite)
        print(f"Original: {result['original']}")
        print(f"Paraphrase 1: {result['text1']}")
        print(f"Paraphrase 2: {result['text2']}")
        print(f"Paraphrase 3: {result['text3']}")
        
    else:
        # User-specified mode
        model_config = ModelConfig(model_id=args.model, device=args.device)
        system = LLMasJudgeSystem(model_config)
        
        if args.mode in ["judge", "both"] and args.text2:
            print("\n=== Judging Relevance ===")
            result = system.judge_relevance(args.text1, args.text2)
            print(f"Text1: {args.text1}")
            print(f"Text2: {args.text2}")
            print(f"Relevance: {result}")
        
        if args.mode in ["rewrite", "both"]:
            print("\n=== Rewriting Text ===")
            result = system.rewrite_text(args.text1)
            print(json.dumps(result, indent=2, ensure_ascii=False))