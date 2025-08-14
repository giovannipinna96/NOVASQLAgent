"""
LLM Summarization and Information Extraction Module.

This module provides a comprehensive system for extracting key information 
and facts from long texts using transformer-based models. It supports
multiple summarization strategies and can extract structured information
from lengthy prompts.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration
)


class SummarizationStrategy(Enum):
    """Enumeration of available summarization strategies."""
    EXTRACTIVE = "extractive"  # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Generate new summary
    HYBRID = "hybrid"  # Combine both approaches
    KEY_FACTS = "key_facts"  # Extract structured facts
    BULLET_POINTS = "bullet_points"  # Generate bullet points


class SummarizationModel(Enum):
    """Enumeration of supported summarization models."""
    PEGASUS_XSUM = "google/pegasus-xsum"
    PEGASUS_CNN = "google/pegasus-cnn_dailymail"
    BART_CNN = "facebook/bart-large-cnn"
    T5_BASE = "google-t5/t5-base"
    T5_SMALL = "google-t5/t5-small"
    LED_BASE = "allenai/led-base-16384"  # For long documents
    BIGBIRD_PEGASUS = "google/bigbird-pegasus-large-arxiv"


@dataclass
class SummarizationConfig:
    """Configuration class for summarization settings."""
    model_name: str = SummarizationModel.PEGASUS_XSUM.value
    strategy: SummarizationStrategy = SummarizationStrategy.ABSTRACTIVE
    max_length: int = 150
    min_length: int = 30
    max_input_length: int = 1024
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    use_pipeline: bool = True  # Use pipeline or direct model calls
    chunk_size: int = 1000  # For processing long texts
    overlap: int = 100  # Overlap between chunks
    extract_entities: bool = False  # Extract named entities
    extract_keywords: bool = False  # Extract key phrases
    custom_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass 
class ExtractionResult:
    """Result container for summarization and extraction operations."""
    original_text: str
    summary: str
    key_facts: List[str] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    strategy_used: str = ""
    chunks_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "original_text_length": len(self.original_text),
            "summary": self.summary,
            "summary_length": len(self.summary),
            "key_facts": self.key_facts,
            "bullet_points": self.bullet_points,
            "entities": self.entities,
            "keywords": self.keywords,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "strategy_used": self.strategy_used,
            "chunks_processed": self.chunks_processed,
            "metadata": self.metadata
        }
    
    def get_structured_output(self) -> Dict[str, Any]:
        """Get structured output with main information."""
        return {
            "summary": self.summary,
            "key_information": {
                "main_facts": self.key_facts[:5],  # Top 5 facts
                "key_points": self.bullet_points[:7],  # Top 7 bullet points
                "important_entities": self.entities[:10]  # Top 10 entities
            },
            "extraction_metadata": {
                "confidence": self.confidence_score,
                "processing_time": f"{self.processing_time:.2f}s",
                "model": self.model_used,
                "strategy": self.strategy_used
            }
        }


class LLMSummarizer:
    """
    Advanced LLM-based text summarization and information extraction system.
    
    This class provides comprehensive text summarization capabilities using 
    transformer models, with support for different strategies and models.
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        """
        Initialize the LLM Summarizer.
        
        Args:
            config: Configuration object with summarization settings
        """
        self.config = config or SummarizationConfig()
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = self._setup_device()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize the model
        self._initialize_model()
    
    def _setup_device(self) -> str:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _initialize_model(self):
        """Initialize the summarization model."""
        try:
            if self.config.use_pipeline:
                self.logger.info(f"Loading pipeline with model: {self.config.model_name}")
                
                # Setup pipeline based on model type
                if "t5" in self.config.model_name.lower():
                    task = "text2text-generation"
                else:
                    task = "summarization"
                
                self.pipeline = pipeline(
                    task=task,
                    model=self.config.model_name,
                    torch_dtype=self.config.torch_dtype,
                    device=0 if self.device == "cuda" else -1
                )
                
            else:
                # Load model and tokenizer separately for more control
                self.logger.info(f"Loading model and tokenizer separately: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                
                # Load appropriate model class based on model name
                if "t5" in self.config.model_name.lower():
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device
                    )
                elif "bart" in self.config.model_name.lower():
                    self.model = BartForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device
                    )
                elif "pegasus" in self.config.model_name.lower():
                    self.model = PegasusForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device
                    )
                else:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device
                    )
                
                self.model.eval()
            
            self.logger.info(f"Model initialized successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def summarize_text(self, text: str, custom_config: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """
        Main method to summarize and extract information from text.
        
        Args:
            text: Input text to summarize
            custom_config: Optional custom configuration overrides
            
        Returns:
            ExtractionResult containing summary and extracted information
        """
        start_time = time.time()
        
        try:
            # Apply custom configuration if provided
            config = self._merge_configs(custom_config) if custom_config else self.config
            
            # Handle long texts by chunking if necessary
            if len(text) > config.max_input_length:
                return self._process_long_text(text, config)
            
            # Generate summary based on strategy
            result = self._generate_summary(text, config)
            result.processing_time = time.time() - start_time
            
            # Extract additional information if requested
            if config.strategy in [SummarizationStrategy.HYBRID, SummarizationStrategy.KEY_FACTS]:
                result = self._extract_key_facts(text, result, config)
            
            if config.strategy == SummarizationStrategy.BULLET_POINTS:
                result = self._generate_bullet_points(text, result, config)
            
            if config.extract_entities:
                result = self._extract_entities(text, result)
            
            if config.extract_keywords:
                result = self._extract_keywords(text, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return ExtractionResult(
                original_text=text,
                summary=f"Error durante la summarizzazione: {str(e)}",
                processing_time=time.time() - start_time,
                model_used=config.model_name,
                strategy_used=config.strategy.value
            )
    
    def _generate_summary(self, text: str, config: SummarizationConfig) -> ExtractionResult:
        """Generate summary using the configured model and strategy."""
        
        # Prepare input text with appropriate prefix for T5-like models
        input_text = self._prepare_input_text(text, config)
        
        try:
            if config.use_pipeline and self.pipeline:
                # Use pipeline for generation
                if "t5" in config.model_name.lower():
                    # T5 models use text2text-generation
                    result = self.pipeline(
                        input_text,
                        max_length=config.max_length,
                        min_length=config.min_length,
                        do_sample=config.do_sample,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        num_beams=config.num_beams,
                        length_penalty=config.length_penalty,
                        early_stopping=config.early_stopping,
                        repetition_penalty=config.repetition_penalty
                    )
                    summary = result[0]['generated_text'] if result else ""
                else:
                    # Other models use summarization pipeline
                    result = self.pipeline(
                        text,
                        max_length=config.max_length,
                        min_length=config.min_length,
                        do_sample=config.do_sample,
                        num_beams=config.num_beams,
                        length_penalty=config.length_penalty,
                        early_stopping=config.early_stopping
                    )
                    summary = result[0]['summary_text'] if result else ""
                    
            else:
                # Use direct model calls for more control
                summary = self._generate_with_model(input_text, config)
            
            # Calculate confidence score (simple heuristic)
            confidence = self._calculate_confidence(text, summary)
            
            return ExtractionResult(
                original_text=text,
                summary=summary,
                confidence_score=confidence,
                model_used=config.model_name,
                strategy_used=config.strategy.value,
                chunks_processed=1
            )
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            raise
    
    def _prepare_input_text(self, text: str, config: SummarizationConfig) -> str:
        """Prepare input text with appropriate prefixes for different models."""
        
        # Custom prompts for different strategies
        if config.strategy == SummarizationStrategy.KEY_FACTS:
            if "t5" in config.model_name.lower():
                return f"extract key facts: {text}"
            else:
                return f"Extract the main facts and key information from this text: {text}"
        
        elif config.strategy == SummarizationStrategy.BULLET_POINTS:
            if "t5" in config.model_name.lower():
                return f"create bullet points: {text}"
            else:
                return f"Create a bullet-point summary of this text: {text}"
        
        elif config.strategy == SummarizationStrategy.EXTRACTIVE:
            return f"Extract the most important sentences: {text}"
        
        else:  # ABSTRACTIVE or HYBRID
            if "t5" in config.model_name.lower():
                return f"summarize: {text}"
            else:
                return text
    
    def _generate_with_model(self, input_text: str, config: SummarizationConfig) -> str:
        """Generate summary using direct model calls."""
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=config.max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=config.max_length,
                min_length=config.min_length,
                num_beams=config.num_beams,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prefix for T5-like models
        if "t5" in config.model_name.lower() and summary.startswith(input_text.split(':')[0]):
            summary = summary.replace(input_text.split(':')[0] + ':', '').strip()
        
        return summary
    
    def _process_long_text(self, text: str, config: SummarizationConfig) -> ExtractionResult:
        """Process long texts by chunking."""
        
        # Split text into chunks with overlap
        chunks = self._chunk_text(text, config.chunk_size, config.overlap)
        
        summaries = []
        total_processing_time = 0
        
        self.logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_result = self._generate_summary(chunk, config)
                summaries.append(chunk_result.summary)
                total_processing_time += chunk_result.processing_time
                
                self.logger.info(f"Processed chunk {i+1}/{len(chunks)}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process chunk {i+1}: {e}")
                summaries.append(f"[Chunk {i+1} processing failed]")
        
        # Combine chunk summaries
        combined_summary = " ".join(summaries)
        
        # If combined summary is still too long, summarize it again
        if len(combined_summary) > config.max_input_length:
            final_result = self._generate_summary(combined_summary, config)
            final_summary = final_result.summary
        else:
            final_summary = combined_summary
        
        return ExtractionResult(
            original_text=text,
            summary=final_summary,
            confidence_score=self._calculate_confidence(text, final_summary),
            processing_time=total_processing_time,
            model_used=config.model_name,
            strategy_used=config.strategy.value,
            chunks_processed=len(chunks)
        )
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def _extract_key_facts(self, text: str, result: ExtractionResult, config: SummarizationConfig) -> ExtractionResult:
        """Extract key facts from the text."""
        
        try:
            # Use a different prompt to extract facts
            fact_config = SummarizationConfig(
                model_name=config.model_name,
                strategy=SummarizationStrategy.KEY_FACTS,
                max_length=200,
                use_pipeline=config.use_pipeline
            )
            
            facts_text = "Extract the main facts: " + text if not config.use_pipeline else text
            
            if config.use_pipeline and self.pipeline:
                if "t5" in config.model_name.lower():
                    facts_result = self.pipeline(
                        f"extract key facts: {text}",
                        max_length=200,
                        num_beams=4
                    )
                    facts_summary = facts_result[0]['generated_text'] if facts_result else ""
                else:
                    # For non-T5 models, use regular summarization with fact-focused prompt
                    facts_summary = result.summary
            else:
                facts_summary = self._generate_with_model(f"extract key facts: {text}", fact_config)
            
            # Extract individual facts (simple approach - split by sentences)
            facts = [fact.strip() for fact in facts_summary.split('.') if fact.strip()]
            result.key_facts = facts[:7]  # Top 7 facts
            
        except Exception as e:
            self.logger.warning(f"Key facts extraction failed: {e}")
            result.key_facts = []
        
        return result
    
    def _generate_bullet_points(self, text: str, result: ExtractionResult, config: SummarizationConfig) -> ExtractionResult:
        """Generate bullet points from the text."""
        
        try:
            if config.use_pipeline and self.pipeline:
                if "t5" in config.model_name.lower():
                    bullet_result = self.pipeline(
                        f"create bullet points: {text}",
                        max_length=250,
                        num_beams=4
                    )
                    bullet_text = bullet_result[0]['generated_text'] if bullet_result else ""
                else:
                    bullet_text = result.summary
            else:
                bullet_text = self._generate_with_model(f"create bullet points: {text}", config)
            
            # Extract bullet points (simple parsing)
            bullet_points = []
            lines = bullet_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    bullet_points.append(line)
                elif line and not bullet_points:  # If no bullet format, treat as sentences
                    bullet_points.extend([f"• {sent.strip()}" for sent in line.split('.') if sent.strip()])
            
            result.bullet_points = bullet_points[:8]  # Top 8 bullet points
            
        except Exception as e:
            self.logger.warning(f"Bullet points generation failed: {e}")
            result.bullet_points = []
        
        return result
    
    def _extract_entities(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract named entities from text (placeholder implementation)."""
        try:
            # This would use NER pipeline or spaCy for entity extraction
            # For now, implement a simple approach
            import re
            
            # Simple entity extraction patterns
            entities = []
            
            # Extract potential person names (capitalized words)
            person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            persons = re.findall(person_pattern, text)
            
            # Extract dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'
            dates = re.findall(date_pattern, text)
            
            # Extract potential organizations (words with Inc, Corp, Ltd, etc.)
            org_pattern = r'\b[A-Z][a-zA-Z\s]*(?:Inc|Corp|Ltd|Company|Organization)\b'
            organizations = re.findall(org_pattern, text)
            
            # Format entities
            for person in persons[:5]:
                entities.append({"text": person, "label": "PERSON", "confidence": 0.7})
            
            for date in dates[:5]:
                entities.append({"text": date, "label": "DATE", "confidence": 0.6})
                
            for org in organizations[:3]:
                entities.append({"text": org, "label": "ORGANIZATION", "confidence": 0.5})
            
            result.entities = entities
            
        except Exception as e:
            self.logger.warning(f"Entity extraction failed: {e}")
            result.entities = []
        
        return result
    
    def _extract_keywords(self, text: str, result: ExtractionResult) -> ExtractionResult:
        """Extract keywords from text (simple implementation)."""
        try:
            import re
            from collections import Counter
            
            # Simple keyword extraction
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
            
            # Extract words
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
            # Filter out stop words and short words
            filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Count frequency and get top keywords
            word_counts = Counter(filtered_words)
            keywords = [word for word, count in word_counts.most_common(10)]
            
            result.keywords = keywords
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            result.keywords = []
        
        return result
    
    def _calculate_confidence(self, original_text: str, summary: str) -> float:
        """Calculate confidence score for the summary (simple heuristic)."""
        try:
            if not summary or not original_text:
                return 0.0
            
            # Simple heuristics for confidence
            summary_length = len(summary.split())
            original_length = len(original_text.split())
            
            # Length ratio should be reasonable (not too short or too long)
            length_ratio = summary_length / original_length if original_length > 0 else 0
            
            # Ideal ratio between 0.1 and 0.4
            if 0.1 <= length_ratio <= 0.4:
                length_score = 1.0
            elif length_ratio < 0.1:
                length_score = length_ratio / 0.1
            else:
                length_score = max(0.5, 1.0 - (length_ratio - 0.4))
            
            # Check for common words (simple overlap)
            original_words = set(original_text.lower().split())
            summary_words = set(summary.lower().split())
            
            overlap_ratio = len(original_words.intersection(summary_words)) / len(summary_words) if summary_words else 0
            overlap_score = min(1.0, overlap_ratio * 2)  # Scale overlap score
            
            # Combine scores
            confidence = (length_score * 0.6 + overlap_score * 0.4)
            return round(confidence, 3)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _merge_configs(self, custom_config: Dict[str, Any]) -> SummarizationConfig:
        """Merge custom configuration with default configuration."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(custom_config)
        
        # Create new config object
        merged_config = SummarizationConfig()
        for key, value in config_dict.items():
            if hasattr(merged_config, key):
                setattr(merged_config, key, value)
        
        return merged_config
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "torch_dtype": str(self.config.torch_dtype),
            "use_pipeline": self.config.use_pipeline,
            "strategy": self.config.strategy.value,
            "max_length": self.config.max_length,
            "chunk_size": self.config.chunk_size
        }


# Factory function for easy instantiation
def create_summarizer(
    model_name: str = "google/pegasus-xsum",
    strategy: str = "abstractive",
    max_length: int = 150,
    device: str = "auto"
) -> LLMSummarizer:
    """
    Factory function to create a summarizer with common settings.
    
    Args:
        model_name: HuggingFace model name
        strategy: Summarization strategy ('abstractive', 'extractive', 'hybrid', 'key_facts', 'bullet_points')
        max_length: Maximum summary length
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Configured LLMSummarizer instance
    """
    
    strategy_enum = SummarizationStrategy(strategy)
    
    config = SummarizationConfig(
        model_name=model_name,
        strategy=strategy_enum,
        max_length=max_length,
        device=device
    )
    
    return LLMSummarizer(config)


# CLI Interface and Testing
if __name__ == "__main__":
    import argparse
    
    def main():
        """Main function for CLI usage."""
        parser = argparse.ArgumentParser(description="LLM Text Summarization and Information Extraction")
        
        parser.add_argument("--text", "-t", required=True, help="Text to summarize")
        parser.add_argument("--model", "-m", default="google/pegasus-xsum", help="Model to use")
        parser.add_argument("--strategy", "-s", choices=["abstractive", "extractive", "hybrid", "key_facts", "bullet_points"], 
                          default="abstractive", help="Summarization strategy")
        parser.add_argument("--max-length", type=int, default=150, help="Maximum summary length")
        parser.add_argument("--output", "-o", help="Output file (JSON format)")
        parser.add_argument("--entities", action="store_true", help="Extract named entities")
        parser.add_argument("--keywords", action="store_true", help="Extract keywords")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        
        args = parser.parse_args()
        
        # Setup logging
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        try:
            # Create config
            config = SummarizationConfig(
                model_name=args.model,
                strategy=SummarizationStrategy(args.strategy),
                max_length=args.max_length,
                extract_entities=args.entities,
                extract_keywords=args.keywords
            )
            
            # Create summarizer
            print("🚀 Initializing LLM Summarizer...")
            summarizer = LLMSummarizer(config)
            
            # Process text
            print("📝 Processing text...")
            result = summarizer.summarize_text(args.text)
            
            # Display results
            print("\n" + "="*50)
            print("📋 SUMMARIZATION RESULTS")
            print("="*50)
            print(f"📄 Original text length: {len(args.text)} characters")
            print(f"📄 Summary length: {len(result.summary)} characters")
            print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
            print(f"🎯 Confidence score: {result.confidence_score:.3f}")
            print(f"🤖 Model used: {result.model_used}")
            print(f"📊 Strategy: {result.strategy_used}")
            
            if result.chunks_processed > 1:
                print(f"🧩 Chunks processed: {result.chunks_processed}")
            
            print("\n📝 SUMMARY:")
            print("-" * 30)
            print(result.summary)
            
            if result.key_facts:
                print(f"\n🔑 KEY FACTS ({len(result.key_facts)}):")
                print("-" * 30)
                for i, fact in enumerate(result.key_facts, 1):
                    print(f"{i}. {fact}")
            
            if result.bullet_points:
                print(f"\n• BULLET POINTS ({len(result.bullet_points)}):")
                print("-" * 30)
                for point in result.bullet_points:
                    print(point)
            
            if result.entities:
                print(f"\n👥 ENTITIES ({len(result.entities)}):")
                print("-" * 30)
                for entity in result.entities:
                    print(f"  {entity['text']} ({entity['label']}) - {entity['confidence']:.2f}")
            
            if result.keywords:
                print(f"\n🏷️  KEYWORDS ({len(result.keywords)}):")
                print("-" * 30)
                print(", ".join(result.keywords))
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                print(f"\n💾 Results saved to: {args.output}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
        
        return 0
    
    # Example usage and testing
    def test_basic_functionality():
        """Test basic functionality with sample text."""
        
        sample_text = """
        L'intelligenza artificiale (IA) è una tecnologia in rapida evoluzione che sta trasformando molti settori. 
        Le applicazioni di machine learning stanno migliorando la medicina, l'automazione industriale e i servizi finanziari. 
        I modelli di linguaggio come GPT e BERT hanno rivoluzionato il processing del linguaggio naturale. 
        Tuttavia, l'IA presenta anche sfide etiche importanti, inclusi bias nei dati e trasparenza degli algoritmi. 
        È fondamentale sviluppare l'IA in modo responsabile, considerando l'impatto sulla società e sul lavoro. 
        Le aziende tech come Google, Microsoft e OpenAI stanno investendo miliardi in ricerca e sviluppo. 
        Il futuro dell'IA dipenderà dalla collaborazione tra ricercatori, industria e regulatori.
        """
        
        print("🧪 Testing LLM Summarizer...")
        
        try:
            # Test with different strategies
            strategies = ["abstractive", "key_facts", "bullet_points"]
            
            for strategy in strategies:
                print(f"\n--- Testing {strategy} strategy ---")
                
                summarizer = create_summarizer(
                    model_name="google-t5/t5-small",  # Use smaller model for testing
                    strategy=strategy,
                    max_length=100
                )
                
                result = summarizer.summarize_text(sample_text)
                
                print(f"Summary: {result.summary}")
                print(f"Confidence: {result.confidence_score:.3f}")
                print(f"Processing time: {result.processing_time:.2f}s")
                
                if result.key_facts:
                    print(f"Key facts: {len(result.key_facts)}")
                
                if result.bullet_points:
                    print(f"Bullet points: {len(result.bullet_points)}")
            
            print("\n✅ Basic functionality test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    # Run CLI or test
    if len(__import__("sys").argv) > 1:
        exit(main())
    else:
        print("Running basic functionality test...")
        test_basic_functionality()