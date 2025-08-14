"""
LLM Selector with Token Probability Analysis

This module implements an advanced LLM Judge that not only looks at the final output
but also analyzes the probability distribution of "yes" and "no" tokens to determine
confidence and potential uncertainty in decisions.

Key Features:
- Analyzes token probabilities for "yes" and "no" 
- Compares token probabilities with final output
- Returns "forse" (maybe) when probabilities disagree with output
- Uses phi-4-mini-instruct for consistent text relevance judgment
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProbabilitySelector:
    """
    Advanced LLM Judge with sophisticated token probability analysis.
    
    Implements Strategy Pattern for different analysis approaches and uses
    Observer Pattern to monitor token-level probabilities for confidence scoring.
    
    Features:
    - Token-level probability extraction and analysis
    - Multi-variant token ID resolution for robust detection
    - Confidence scoring with uncertainty quantification
    - Disagreement detection between output and probabilities
    - Batch processing with individual probability tracking
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM Probability Selector.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference ("cpu", "cuda", etc.). Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Cache token IDs for "yes" and "no"
        self.yes_token_ids = self._get_token_variants("yes")
        self.no_token_ids = self._get_token_variants("no")
        
        logger.info(f"Yes token IDs: {self.yes_token_ids}")
        logger.info(f"No token IDs: {self.no_token_ids}")
        
        # System prompt for relevance judgment
        self.system_prompt = (
            "You are a text relevance judge. Your task is to determine if one text "
            "is relevant to another text. You must respond with only 'yes' or 'no'. "
            "Do not provide explanations or additional text."
        )
    
    def _load_model(self):
        """Load the model and tokenizer for probability analysis."""
        try:
            logger.info(f"Loading LLM Selector model: {self.model_id}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()  # Set to evaluation mode
            logger.info("LLM Selector model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM Selector model: {str(e)}")
            raise
    
    def _get_token_variants(self, word: str) -> List[int]:
        """
        Get all possible token ID variants for a word (considering different tokenizations).
        
        Args:
            word: The word to get token IDs for
            
        Returns:
            List of token IDs that could represent the word
        """
        variants = []
        
        # Common variations to check
        word_variants = [
            word.lower(),
            word.upper(), 
            word.capitalize(),
            f" {word.lower()}",  # With leading space
            f" {word.upper()}",
            f" {word.capitalize()}",
            f"{word.lower()}.",  # With punctuation
            f"{word.upper()}.",
            f"{word.capitalize()}."
        ]
        
        for variant in word_variants:
            try:
                # Encode the variant and get token IDs
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                variants.extend(token_ids)
            except:
                continue
        
        # Remove duplicates and return
        return list(set(variants))
    
    def _extract_token_probabilities(self, logits: torch.Tensor) -> Dict[str, float]:
        """
        Extract probabilities for "yes" and "no" tokens from logits.
        
        Args:
            logits: Model logits tensor of shape (batch_size, vocab_size)
            
        Returns:
            Dictionary with "yes_prob" and "no_prob" values
        """
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Sum probabilities for all "yes" variants
        yes_prob = 0.0
        for token_id in self.yes_token_ids:
            if token_id < probabilities.shape[-1]:
                yes_prob += probabilities[0, token_id].item()
        
        # Sum probabilities for all "no" variants  
        no_prob = 0.0
        for token_id in self.no_token_ids:
            if token_id < probabilities.shape[-1]:
                no_prob += probabilities[0, token_id].item()
        
        return {
            "yes_prob": yes_prob,
            "no_prob": no_prob
        }
    
    def _generate_with_scores(self, prompt: str) -> Tuple[str, Dict[str, float]]:
        """
        Generate response and extract token probabilities.
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Tuple of (generated_text, probability_scores)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded properly")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation configuration
        generation_config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,  # Deterministic for probability analysis
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True  # This is crucial for getting token probabilities!
        )
        
        with torch.no_grad():
            # Generate with scores
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            
            # Extract generated text
            generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Extract token probabilities from the first generated token
            # outputs.scores contains logits for each generation step
            if outputs.scores and len(outputs.scores) > 0:
                first_token_logits = outputs.scores[0]  # Logits for first generated token
                token_probs = self._extract_token_probabilities(first_token_logits)
            else:
                logger.warning("No scores available from generation")
                token_probs = {"yes_prob": 0.0, "no_prob": 0.0}
        
        return generated_text, token_probs
    
    def judge_with_probabilities(self, text1: str, text2: str) -> Dict[str, Union[str, float]]:
        """
        Judge text relevance with detailed probability analysis.
        
        Args:
            text1: First text to compare
            text2: Second text to compare against
            
        Returns:
            Dictionary containing judgment, probabilities, and confidence analysis
        """
        # Construct the prompt
        prompt = f"""System: {self.system_prompt}

Question: Is the following text relevant to the reference text?

Reference text: {text2}

Text to evaluate: {text1}

Answer (yes/no):"""
        
        try:
            # Generate response with probability analysis
            generated_text, token_probs = self._generate_with_scores(prompt)
            
            # Extract final output decision
            final_output = "no"  # default
            if "yes" in generated_text.lower():
                final_output = "yes"
            elif "no" in generated_text.lower():
                final_output = "no"
            
            # Determine which token has higher probability
            yes_prob = token_probs["yes_prob"]
            no_prob = token_probs["no_prob"]
            
            prob_based_decision = "yes" if yes_prob > no_prob else "no"
            
            # Calculate confidence metrics
            total_prob = yes_prob + no_prob
            confidence = max(yes_prob, no_prob) / total_prob if total_prob > 0 else 0.0
            prob_difference = abs(yes_prob - no_prob)
            
            # Determine final judgment
            final_judgment = final_output
            
            # Check for disagreement between probability-based decision and final output
            if final_output != prob_based_decision:
                # If probabilities disagree with output, return "forse" (maybe)
                final_judgment = "forse"
                logger.info(f"Disagreement detected: Output='{final_output}', Probabilities favor='{prob_based_decision}'")
            
            # Low confidence threshold - also return "forse" if very uncertain
            elif confidence < 0.6 or prob_difference < 0.1:
                final_judgment = "forse"
                logger.info(f"Low confidence detected: confidence={confidence:.3f}, prob_diff={prob_difference:.3f}")
            
            result = {
                "judgment": final_judgment,
                "final_output": final_output,
                "prob_based_decision": prob_based_decision,
                "yes_probability": yes_prob,
                "no_probability": no_prob,
                "confidence": confidence,
                "prob_difference": prob_difference,
                "generated_text": generated_text,
                "agreement": final_output == prob_based_decision
            }
            
            # Log detailed analysis
            logger.info(f"Analysis - Output: {final_output}, Yes prob: {yes_prob:.4f}, No prob: {no_prob:.4f}, Final: {final_judgment}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during probability judgment: {str(e)}")
            return {
                "judgment": "forse",
                "final_output": "error",
                "prob_based_decision": "error",
                "yes_probability": 0.0,
                "no_probability": 0.0,
                "confidence": 0.0,
                "prob_difference": 0.0,
                "generated_text": "Error occurred",
                "agreement": False,
                "error": str(e)
            }
    
    def simple_judge(self, text1: str, text2: str) -> str:
        """
        Simple interface that returns just the judgment.
        
        Args:
            text1: First text to compare
            text2: Second text to compare against
            
        Returns:
            "yes", "no", or "forse" based on analysis
        """
        result = self.judge_with_probabilities(text1, text2)
        return result["judgment"]
    
    def batch_judge(self, text_pairs: List[Tuple[str, str]]) -> List[Dict[str, Union[str, float]]]:
        """
        Judge multiple text pairs with probability analysis.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            
        Returns:
            List of judgment dictionaries
        """
        results = []
        for i, (text1, text2) in enumerate(text_pairs):
            logger.info(f"Processing pair {i+1}/{len(text_pairs)}")
            result = self.judge_with_probabilities(text1, text2)
            results.append(result)
        return results


class LLMSelectorSystem:
    """
    High-level interface for the LLM Selector with probability analysis.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM Selector System.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference
        """
        logger.info("Initializing LLM Selector System with Probability Analysis")
        self.selector = LLMProbabilitySelector(model_id, device)
        logger.info("LLM Selector System initialized successfully")
    
    def judge(self, text1: str, text2: str, detailed: bool = False) -> Union[str, Dict[str, Union[str, float]]]:
        """
        Judge text relevance.
        
        Args:
            text1: First text to compare
            text2: Second text to compare against
            detailed: If True, return detailed analysis; if False, return simple judgment
            
        Returns:
            Simple judgment string or detailed analysis dictionary
        """
        if detailed:
            return self.selector.judge_with_probabilities(text1, text2)
        else:
            return self.selector.simple_judge(text1, text2)
    
    def batch_judge(self, text_pairs: List[Tuple[str, str]], detailed: bool = False) -> List[Union[str, Dict]]:
        """
        Judge multiple text pairs.
        
        Args:
            text_pairs: List of (text1, text2) tuples
            detailed: If True, return detailed analysis; if False, return simple judgments
            
        Returns:
            List of judgments
        """
        if detailed:
            return self.selector.batch_judge(text_pairs)
        else:
            results = self.selector.batch_judge(text_pairs)
            return [result["judgment"] for result in results]


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Selector with Token Probability Analysis")
    parser.add_argument("--text1", type=str, help="First text (text to evaluate)")
    parser.add_argument("--text2", type=str, help="Second text (reference text)")
    parser.add_argument("--model", type=str, default="microsoft/phi-4-mini-instruct",
                       help="Model ID to use")
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed probability analysis")
    parser.add_argument("--batch", type=str, 
                       help="Path to JSON file with text pairs for batch processing")
    
    args = parser.parse_args()
    
    # Initialize system
    system = LLMSelectorSystem(args.model, args.device)
    
    if args.batch:
        # Batch processing mode
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            text_pairs = [(item['text1'], item['text2']) for item in batch_data]
            results = system.batch_judge(text_pairs, detailed=args.detailed)
            
            print("\n=== BATCH PROCESSING RESULTS ===")
            for i, result in enumerate(results):
                if args.detailed:
                    print(f"\nPair {i+1}:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(f"Pair {i+1}: {result}")
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            
    elif args.text1 and args.text2:
        # Single pair mode
        print(f"\n=== ANALYZING TEXT RELEVANCE ===")
        print(f"Text 1: {args.text1}")
        print(f"Text 2: {args.text2}")
        print("=" * 50)
        
        result = system.judge(args.text1, args.text2, detailed=True)
        
        print(f"\nFINAL JUDGMENT: {result['judgment'].upper()}")
        print(f"Generated Output: '{result['generated_text']}'")
        print(f"Final Output Decision: {result['final_output']}")
        print(f"Probability-based Decision: {result['prob_based_decision']}")
        print(f"Yes Token Probability: {result['yes_probability']:.4f}")
        print(f"No Token Probability: {result['no_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probability Difference: {result['prob_difference']:.4f}")
        print(f"Agreement: {'✓' if result['agreement'] else '✗'}")
        
        if not args.detailed:
            simple_result = system.judge(args.text1, args.text2, detailed=False)
            print(f"\nSimple Result: {simple_result}")
    
    else:
        # Demo mode
        print("\n=== DEMO MODE ===")
        print("Running demonstrations with example text pairs...\n")
        
        demo_pairs = [
            # Related texts
            ("The weather is sunny today.", "It's a beautiful day outside."),
            # Unrelated texts  
            ("The cat is sleeping.", "Python is a programming language."),
            # Somewhat related
            ("I love reading books.", "The library has many interesting novels."),
            # Edge case - similar topics but different focus
            ("Machine learning algorithms are powerful.", "Artificial intelligence will change the world."),
        ]
        
        for i, (text1, text2) in enumerate(demo_pairs, 1):
            print(f"\n--- Example {i} ---")
            print(f"Text 1: {text1}")
            print(f"Text 2: {text2}")
            
            result = system.judge(text1, text2, detailed=True)
            
            print(f"Judgment: {result['judgment'].upper()}")
            print(f"Output: {result['final_output']} | Probabilities favor: {result['prob_based_decision']}")
            print(f"Yes: {result['yes_probability']:.3f} | No: {result['no_probability']:.3f} | Confidence: {result['confidence']:.3f}")
            print(f"Agreement: {'✓' if result['agreement'] else '✗'}")