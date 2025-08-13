"""
LLM as Judge and Text Rewriter Module

This module implements two specialized LLMs using phi-4-mini-instruct:
1. LLMJudge: Evaluates text relevance with simple yes/no responses
2. LLMRewriter: Generates 3 paraphrases of input text in JSON format

Both models use the HuggingFace Transformers pipeline for efficient inference.
"""

import json
import torch
from typing import Dict, List, Optional, Union
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


class LLMJudge:
    """
    LLM Judge for text relevance evaluation.
    
    This class uses phi-4-mini-instruct to determine if one text is relevant
    to another text, responding with simple "yes" or "no" answers.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM Judge.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference ("cpu", "cuda", etc.). Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._load_model()
        
        # System prompt for relevance judgment
        self.system_prompt = (
            "You are a text relevance judge. Your task is to determine if one text "
            "is relevant to another text. You must respond with only 'yes' or 'no'. "
            "Do not provide explanations or additional text."
        )
    
    def _load_model(self):
        """Load the model and create the pipeline."""
        try:
            logger.info(f"Loading LLM Judge model: {self.model_id}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                return_full_text=False,
                max_new_tokens=10,  # Short response needed
                do_sample=False,    # Deterministic output
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            logger.info("LLM Judge model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM Judge model: {str(e)}")
            raise
    
    def judge_relevance(self, text1: str, text2: str) -> str:
        """
        Judge if text1 is relevant to text2.
        
        Args:
            text1: First text to compare
            text2: Second text to compare against
            
        Returns:
            "yes" if texts are relevant, "no" otherwise
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded properly")
        
        # Construct the prompt
        prompt = f"""System: {self.system_prompt}

Question: Is the following text relevant to the reference text?

Reference text: {text2}

Text to evaluate: {text1}

Answer (yes/no):"""
        
        try:
            # Generate response
            result = self.pipeline(prompt)
            response = result[0]["generated_text"].strip().lower()
            
            # Extract yes/no from response
            if "yes" in response:
                return "yes"
            elif "no" in response:
                return "no"
            else:
                # Default to "no" if unclear
                logger.warning(f"Unclear response from judge: {response}")
                return "no"
                
        except Exception as e:
            logger.error(f"Error during judgment: {str(e)}")
            return "no"


class LLMRewriter:
    """
    LLM Rewriter for generating text paraphrases.
    
    This class uses phi-4-mini-instruct to generate 3 different paraphrases
    of the input text, returning results in JSON format.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM Rewriter.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference ("cpu", "cuda", etc.). Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._load_model()
        
        # System prompt for text rewriting
        self.system_prompt = (
            "You are a text rewriter. Your task is to create exactly 3 different "
            "paraphrases of the given text. Maintain the same meaning but vary the "
            "wording and structure. Return your response as valid JSON with keys: "
            "'original', 'text1', 'text2', 'text3'."
        )
    
    def _load_model(self):
        """Load the model and create the pipeline."""
        try:
            logger.info(f"Loading LLM Rewriter model: {self.model_id}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                return_full_text=False,
                max_new_tokens=300,  # Enough for 3 paraphrases
                do_sample=True,     # Creative generation
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            logger.info("LLM Rewriter model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM Rewriter model: {str(e)}")
            raise
    
    def rewrite_text(self, text: str) -> Dict[str, str]:
        """
        Generate 3 paraphrases of the input text.
        
        Args:
            text: Original text to paraphrase
            
        Returns:
            Dictionary with original text and 3 paraphrases
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded properly")
        
        # Construct the prompt
        prompt = f"""System: {self.system_prompt}

Original text: {text}

Generate 3 paraphrases in JSON format:"""
        
        try:
            # Generate response
            result = self.pipeline(prompt)
            response = result[0]["generated_text"].strip()
            
            # Try to extract JSON from response
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
                        return parsed_json
                
                # If JSON parsing fails, create structured response manually
                logger.warning("Failed to parse JSON, creating fallback response")
                return self._create_fallback_response(text, response)
                
            except json.JSONDecodeError:
                logger.warning("JSON decode error, creating fallback response")
                return self._create_fallback_response(text, response)
                
        except Exception as e:
            logger.error(f"Error during rewriting: {str(e)}")
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


class LLMasJudgeSystem:
    """
    Combined system that provides both judging and rewriting capabilities.
    
    This class manages both LLMJudge and LLMRewriter instances, providing
    a unified interface for text relevance evaluation and paraphrasing.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the combined LLM system.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference
        """
        self.model_id = model_id
        self.device = device
        
        logger.info("Initializing LLM as Judge System")
        self.judge = LLMJudge(model_id, device)
        self.rewriter = LLMRewriter(model_id, device)
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
        
        system = LLMasJudgeSystem(args.model, args.device)
        
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
        system = LLMasJudgeSystem(args.model, args.device)
        
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