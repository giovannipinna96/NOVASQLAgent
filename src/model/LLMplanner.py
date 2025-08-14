"""
LLM Request Planning Module.

This module provides a comprehensive system for breaking down complex requests 
into simpler, more manageable sub-tasks using transformer-based language models.
The system generates structured JSON output with sequential steps.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)


class PlanningStrategy(Enum):
    """Enumeration of available planning strategies."""
    SEQUENTIAL = "sequential"  # Break into sequential steps
    HIERARCHICAL = "hierarchical"  # Create hierarchical breakdown
    DEPENDENCY_BASED = "dependency_based"  # Based on dependencies
    CATEGORY_BASED = "category_based"  # Group by categories
    PRIORITY_BASED = "priority_based"  # Order by priority


class PlanningModel(Enum):
    """Enumeration of supported planning models."""
    T5_BASE = "google-t5/t5-base"
    T5_LARGE = "google-t5/t5-large"
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    PHI_4_MINI = "microsoft/phi-4-mini-instruct"
    GEMMA_2B = "google/gemma-2b"
    QWEN_2_5 = "Qwen/Qwen2.5-1.5B-Instruct"
    COHERE_COMMAND_R = "CohereForAI/c4ai-command-r-v01"


@dataclass
class PlanningConfig:
    """Configuration class for planning settings."""
    model_name: str = PlanningModel.T5_BASE.value
    strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL
    max_steps: int = 10
    min_steps: int = 2
    max_length: int = 512
    max_input_length: int = 1024
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 3
    do_sample: bool = True
    repetition_penalty: float = 1.1
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    use_pipeline: bool = True
    include_dependencies: bool = False
    include_priorities: bool = False
    include_categories: bool = False
    include_complexity: bool = False
    use_chat_template: bool = True  # Use chat template for instruct models
    custom_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PlanningResult:
    """Result container for planning operations."""
    original_request: str
    steps: Dict[str, str] = field(default_factory=dict)  # step1, step2, etc.
    strategy_used: str = ""
    total_steps: int = 0
    complexity_score: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    confidence_score: float = 0.0
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # step -> [required steps]
    priorities: Dict[str, int] = field(default_factory=dict)  # step -> priority level
    categories: Dict[str, str] = field(default_factory=dict)  # step -> category
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON format as requested."""
        result_dict = {
            "original": self.original_request,
            **self.steps  # step1, step2, etc.
        }
        
        # Add metadata if available
        if self.dependencies or self.priorities or self.categories:
            result_dict["metadata"] = {}
            if self.dependencies:
                result_dict["metadata"]["dependencies"] = self.dependencies
            if self.priorities:
                result_dict["metadata"]["priorities"] = self.priorities
            if self.categories:
                result_dict["metadata"]["categories"] = self.categories
        
        return json.dumps(result_dict, indent=indent, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "original_request": self.original_request,
            "steps": self.steps,
            "strategy_used": self.strategy_used,
            "total_steps": self.total_steps,
            "complexity_score": self.complexity_score,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "confidence_score": self.confidence_score,
            "dependencies": self.dependencies,
            "priorities": self.priorities,
            "categories": self.categories,
            "metadata": self.metadata
        }
    
    def get_formatted_plan(self) -> str:
        """Get a formatted plan as readable text."""
        lines = [f"📋 Original Request: {self.original_request}", "", "📝 Planned Steps:"]
        
        for step_key in sorted(self.steps.keys(), key=lambda x: int(x.replace('step', ''))):
            step_num = step_key.replace('step', '')
            step_text = self.steps[step_key]
            
            # Add priority and category if available
            annotations = []
            if step_key in self.priorities:
                annotations.append(f"Priority: {self.priorities[step_key]}")
            if step_key in self.categories:
                annotations.append(f"Category: {self.categories[step_key]}")
            
            annotation_text = f" ({', '.join(annotations)})" if annotations else ""
            lines.append(f"{step_num}. {step_text}{annotation_text}")
            
            # Add dependencies if available
            if step_key in self.dependencies and self.dependencies[step_key]:
                deps = ", ".join([dep.replace('step', '') for dep in self.dependencies[step_key]])
                lines.append(f"   └─ Depends on: {deps}")
        
        return "\n".join(lines)


class LLMPlanner:
    """
    Advanced LLM-based request planning and decomposition system.
    
    This class provides comprehensive request planning capabilities using 
    transformer models to break down complex requests into manageable steps.
    """
    
    def __init__(self, config: Optional[PlanningConfig] = None):
        """
        Initialize the LLM Planner.
        
        Args:
            config: Configuration object with planning settings
        """
        self.config = config or PlanningConfig()
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
        """Initialize the planning model."""
        try:
            if self.config.use_pipeline:
                self.logger.info(f"Loading pipeline with model: {self.config.model_name}")
                
                # Setup pipeline based on model type
                if "t5" in self.config.model_name.lower():
                    task = "text2text-generation"
                else:
                    task = "text-generation"
                
                self.pipeline = pipeline(
                    task=task,
                    model=self.config.model_name,
                    torch_dtype=self.config.torch_dtype,
                    device=0 if self.device == "cuda" else -1,
                    trust_remote_code=True
                )
                
                # Load tokenizer for chat template support
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name, 
                    trust_remote_code=True
                )
                
                # Ensure tokenizer has pad token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
            else:
                # Load model and tokenizer separately for more control
                self.logger.info(f"Loading model and tokenizer separately: {self.config.model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load appropriate model class
                if "t5" in self.config.model_name.lower():
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=self.config.torch_dtype,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                
                self.model.eval()
            
            self.logger.info(f"Model initialized successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def plan_request(self, complex_request: str, custom_config: Optional[Dict[str, Any]] = None) -> PlanningResult:
        """
        Main method to break down a complex request into simple steps.
        
        Args:
            complex_request: The complex request to break down
            custom_config: Optional custom configuration overrides
            
        Returns:
            PlanningResult containing the breakdown and metadata
        """
        start_time = time.time()
        
        try:
            # Apply custom configuration if provided
            config = self._merge_configs(custom_config) if custom_config else self.config
            
            # Generate plan based on strategy
            result = self._generate_plan(complex_request, config)
            result.processing_time = time.time() - start_time
            
            # Add additional information if requested
            if config.include_dependencies:
                result = self._analyze_dependencies(result, config)
            
            if config.include_priorities:
                result = self._analyze_priorities(result, config)
                
            if config.include_categories:
                result = self._analyze_categories(result, config)
                
            if config.include_complexity:
                result.complexity_score = self._calculate_complexity_score(complex_request, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            return PlanningResult(
                original_request=complex_request,
                steps={"step1": f"Error during planning: {str(e)}"},
                processing_time=time.time() - start_time,
                model_used=config.model_name,
                strategy_used=config.strategy.value,
                total_steps=1
            )
    
    def _generate_plan(self, request: str, config: PlanningConfig) -> PlanningResult:
        """Generate plan using the configured model and strategy."""
        
        # Create prompt based on strategy
        prompt = self._create_planning_prompt(request, config)
        
        try:
            if config.use_pipeline and self.pipeline:
                # Use pipeline for generation
                if "t5" in config.model_name.lower():
                    # T5 models use text2text-generation
                    result = self.pipeline(
                        prompt,
                        max_length=config.max_length,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        num_beams=config.num_beams,
                        do_sample=config.do_sample,
                        repetition_penalty=config.repetition_penalty
                    )
                    plan_text = result[0]['generated_text'] if result else ""
                else:
                    # Other models use text-generation
                    # Use chat template if available and enabled
                    if config.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                        messages = self._create_chat_messages(request, config)
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    else:
                        formatted_prompt = prompt
                    
                    result = self.pipeline(
                        formatted_prompt,
                        max_new_tokens=config.max_length,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        do_sample=config.do_sample,
                        repetition_penalty=config.repetition_penalty,
                        return_full_text=False
                    )
                    plan_text = result[0]['generated_text'] if result else ""
                    
            else:
                # Use direct model calls for more control
                plan_text = self._generate_with_model(prompt, config)
            
            # Parse the generated plan
            steps = self._parse_plan_text(plan_text, config)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(request, plan_text, steps)
            
            return PlanningResult(
                original_request=request,
                steps=steps,
                strategy_used=config.strategy.value,
                total_steps=len(steps),
                model_used=config.model_name,
                confidence_score=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Plan generation failed: {e}")
            # Fallback to simple breakdown
            return self._generate_fallback_plan(request, config)
    
    def _create_planning_prompt(self, request: str, config: PlanningConfig) -> str:
        """Create planning prompt based on strategy and configuration."""
        
        # Check for custom prompts
        if config.strategy.value in config.custom_prompts:
            base_prompt = config.custom_prompts[config.strategy.value]
            return base_prompt.format(request=request, max_steps=config.max_steps)
        
        # Default prompts based on strategy
        if config.strategy == PlanningStrategy.SEQUENTIAL:
            if "t5" in config.model_name.lower():
                prompt = f"""break down task: {request}"""
            else:
                prompt = f"""Break down this complex request into {config.max_steps} or fewer sequential steps. 
Return the result as a numbered list where each step is a simple, actionable task.

Request: {request}

Steps:
1."""
        
        elif config.strategy == PlanningStrategy.HIERARCHICAL:
            if "t5" in config.model_name.lower():
                prompt = f"""create hierarchical breakdown: {request}"""
            else:
                prompt = f"""Create a hierarchical breakdown of this request into main categories and sub-tasks.
Organize the steps in a logical hierarchy with at most {config.max_steps} total steps.

Request: {request}

Hierarchical Breakdown:
1."""
        
        elif config.strategy == PlanningStrategy.DEPENDENCY_BASED:
            if "t5" in config.model_name.lower():
                prompt = f"""identify task dependencies: {request}"""
            else:
                prompt = f"""Break down this request into steps considering dependencies between tasks.
Identify which steps must be completed before others can begin.

Request: {request}

Steps with dependencies:
1."""
        
        elif config.strategy == PlanningStrategy.CATEGORY_BASED:
            if "t5" in config.model_name.lower():
                prompt = f"""categorize task steps: {request}"""
            else:
                prompt = f"""Break down this request into steps organized by categories or functional areas.
Group related tasks together and organize logically.

Request: {request}

Categorized Steps:
1."""
        
        elif config.strategy == PlanningStrategy.PRIORITY_BASED:
            if "t5" in config.model_name.lower():
                prompt = f"""prioritize task steps: {request}"""
            else:
                prompt = f"""Break down this request into steps ordered by priority and importance.
Start with the most critical tasks first.

Request: {request}

Prioritized Steps:
1."""
        
        else:
            # Default to sequential
            prompt = f"""Break down this request into simple, actionable steps:

{request}

Steps:
1."""
        
        return prompt
    
    def _create_chat_messages(self, request: str, config: PlanningConfig) -> List[Dict[str, str]]:
        """Create chat messages for instruct models."""
        
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that breaks down complex requests into simple, actionable steps. Always respond with a clear, numbered list of steps."
        }
        
        # Create user message based on strategy
        if config.strategy == PlanningStrategy.SEQUENTIAL:
            user_content = f"Break down this complex request into {config.max_steps} or fewer sequential steps: {request}"
        elif config.strategy == PlanningStrategy.HIERARCHICAL:
            user_content = f"Create a hierarchical breakdown of this request: {request}"
        elif config.strategy == PlanningStrategy.DEPENDENCY_BASED:
            user_content = f"Break down this request into steps, considering dependencies: {request}"
        elif config.strategy == PlanningStrategy.CATEGORY_BASED:
            user_content = f"Break down this request into categorized steps: {request}"
        elif config.strategy == PlanningStrategy.PRIORITY_BASED:
            user_content = f"Break down this request into prioritized steps: {request}"
        else:
            user_content = f"Break down this request into simple steps: {request}"
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        return [system_message, user_message]
    
    def _generate_with_model(self, prompt: str, config: PlanningConfig) -> str:
        """Generate plan using direct model calls."""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=config.max_input_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate plan
        with torch.no_grad():
            if "t5" in config.model_name.lower():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    num_beams=config.num_beams,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty
                )
            else:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=config.max_length,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        # Decode output
        if "t5" in config.model_name.lower():
            plan_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal LM, extract only the generated part
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            plan_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return plan_text
    
    def _parse_plan_text(self, plan_text: str, config: PlanningConfig) -> Dict[str, str]:
        """Parse generated plan text into structured steps."""
        
        steps = {}
        
        try:
            # Clean the text
            plan_text = plan_text.strip()
            
            # Try to extract numbered steps using various patterns
            patterns = [
                r'(\\d+)\\.(.*?)(?=\\n\\d+\\.|$)',  # "1. Step text"
                r'(\\d+)\\)(.*?)(?=\\n\\d+\\)|$)',   # "1) Step text"
                r'Step\\s+(\\d+):(.*?)(?=\\nStep\\s+\\d+:|$)',  # "Step 1: text"
                r'(\\d+)\\s*[-–—](.*?)(?=\\n\\d+\\s*[-–—]|$)',  # "1 - Step text"
            ]
            
            step_matches = []
            for pattern in patterns:
                matches = re.findall(pattern, plan_text, re.MULTILINE | re.DOTALL)
                if matches:
                    step_matches = matches
                    break
            
            if not step_matches:
                # Fallback: split by lines and look for any line that starts with a number
                lines = plan_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and re.match(r'^\\d+', line):
                        # Extract number and text
                        parts = re.split(r'^(\\d+)[.)\\-–—\\s]+', line, maxsplit=1)
                        if len(parts) >= 3:
                            step_num = int(parts[1])
                            step_text = parts[2].strip()
                            if step_text:
                                step_matches.append((str(step_num), step_text))
            
            # Convert matches to steps dictionary
            if step_matches:
                for step_num, step_text in step_matches:
                    step_text = step_text.strip()
                    # Remove trailing punctuation and extra whitespace
                    step_text = re.sub(r'\\s+', ' ', step_text)
                    step_text = step_text.strip('.,;:')
                    
                    if step_text:
                        step_key = f"step{step_num}"
                        steps[step_key] = step_text
                        
                        # Limit to max_steps
                        if len(steps) >= config.max_steps:
                            break
            
            # If no structured steps found, create a fallback
            if not steps:
                # Split by sentences or lines and create steps
                sentences = re.split(r'[.!?]+|\\n+', plan_text)
                step_num = 1
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10:  # Ignore very short fragments
                        steps[f"step{step_num}"] = sentence
                        step_num += 1
                        if step_num > config.max_steps:
                            break
            
            # Ensure we have at least min_steps
            if len(steps) < config.min_steps:
                # Add generic steps if needed
                while len(steps) < config.min_steps:
                    step_num = len(steps) + 1
                    steps[f"step{step_num}"] = f"Complete remaining tasks from the original request"
                    if step_num >= config.max_steps:
                        break
            
        except Exception as e:
            self.logger.warning(f"Failed to parse plan text: {e}")
            # Final fallback
            steps = {
                "step1": "Analyze the request and identify main requirements",
                "step2": "Execute the main task or implement the solution",
                "step3": "Review and validate the results"
            }
        
        return steps
    
    def _generate_fallback_plan(self, request: str, config: PlanningConfig) -> PlanningResult:
        """Generate a fallback plan using simple heuristics."""
        
        request_lower = request.lower()
        steps = {}
        
        # Simple analysis-based planning
        if any(word in request_lower for word in ["create", "build", "develop", "implement", "design"]):
            steps = {
                "step1": "Define requirements and specifications",
                "step2": "Design the solution architecture",
                "step3": "Implement the core functionality",
                "step4": "Test and validate the implementation",
                "step5": "Deploy and finalize the solution"
            }
        elif any(word in request_lower for word in ["analyze", "research", "investigate", "study"]):
            steps = {
                "step1": "Gather and review relevant information",
                "step2": "Perform detailed analysis",
                "step3": "Identify key findings and insights",
                "step4": "Prepare summary and recommendations"
            }
        elif any(word in request_lower for word in ["fix", "solve", "debug", "troubleshoot"]):
            steps = {
                "step1": "Identify and understand the problem",
                "step2": "Research potential solutions",
                "step3": "Implement the fix or solution",
                "step4": "Test and verify the solution works"
            }
        else:
            # Generic breakdown
            steps = {
                "step1": "Break down the request into components",
                "step2": "Address each component systematically",
                "step3": "Integrate results and finalize"
            }
        
        # Limit steps according to config
        limited_steps = {}
        for i, (key, value) in enumerate(steps.items(), 1):
            if i <= config.max_steps:
                limited_steps[key] = value
            else:
                break
        
        return PlanningResult(
            original_request=request,
            steps=limited_steps,
            strategy_used=config.strategy.value + "_fallback",
            total_steps=len(limited_steps),
            model_used=config.model_name + "_fallback",
            confidence_score=0.5
        )
    
    def _analyze_dependencies(self, result: PlanningResult, config: PlanningConfig) -> PlanningResult:
        """Analyze dependencies between steps (simplified implementation)."""
        try:
            dependencies = {}
            step_keys = sorted(result.steps.keys(), key=lambda x: int(x.replace('step', '')))
            
            for i, step_key in enumerate(step_keys):
                step_text = result.steps[step_key].lower()
                deps = []
                
                # Simple dependency detection based on keywords
                if i > 0:  # Steps after the first can have dependencies
                    # Look for dependency keywords
                    if any(word in step_text for word in ["implement", "build", "create", "develop"]):
                        if any(word in result.steps[step_keys[0]].lower() for word in ["define", "plan", "design"]):
                            deps.append(step_keys[0])
                    
                    if any(word in step_text for word in ["test", "validate", "verify"]):
                        # Testing depends on implementation
                        for j in range(i):
                            if any(word in result.steps[step_keys[j]].lower() for word in ["implement", "build", "create"]):
                                deps.append(step_keys[j])
                                break
                    
                    if any(word in step_text for word in ["deploy", "finalize", "complete"]):
                        # Final steps depend on testing
                        for j in range(i):
                            if any(word in result.steps[step_keys[j]].lower() for word in ["test", "validate"]):
                                deps.append(step_keys[j])
                                break
                
                if deps:
                    dependencies[step_key] = deps
            
            result.dependencies = dependencies
            
        except Exception as e:
            self.logger.warning(f"Dependency analysis failed: {e}")
            result.dependencies = {}
        
        return result
    
    def _analyze_priorities(self, result: PlanningResult, config: PlanningConfig) -> PlanningResult:
        """Analyze priorities of steps (simplified implementation)."""
        try:
            priorities = {}
            
            for step_key, step_text in result.steps.items():
                step_text_lower = step_text.lower()
                
                # High priority (1) - Planning and critical steps
                if any(word in step_text_lower for word in ["define", "plan", "design", "analyze", "understand"]):
                    priorities[step_key] = 1
                # Medium priority (2) - Implementation steps  
                elif any(word in step_text_lower for word in ["implement", "build", "create", "develop", "execute"]):
                    priorities[step_key] = 2
                # Lower priority (3) - Testing and finalization
                elif any(word in step_text_lower for word in ["test", "validate", "review", "deploy", "finalize"]):
                    priorities[step_key] = 3
                else:
                    priorities[step_key] = 2  # Default to medium priority
            
            result.priorities = priorities
            
        except Exception as e:
            self.logger.warning(f"Priority analysis failed: {e}")
            result.priorities = {}
        
        return result
    
    def _analyze_categories(self, result: PlanningResult, config: PlanningConfig) -> PlanningResult:
        """Analyze categories for steps (simplified implementation)."""
        try:
            categories = {}
            
            for step_key, step_text in result.steps.items():
                step_text_lower = step_text.lower()
                
                # Categorize based on keywords
                if any(word in step_text_lower for word in ["define", "plan", "design", "analyze", "research"]):
                    categories[step_key] = "Planning"
                elif any(word in step_text_lower for word in ["implement", "build", "create", "develop", "code"]):
                    categories[step_key] = "Implementation"
                elif any(word in step_text_lower for word in ["test", "validate", "verify", "check"]):
                    categories[step_key] = "Testing"
                elif any(word in step_text_lower for word in ["deploy", "release", "finalize", "complete"]):
                    categories[step_key] = "Deployment"
                elif any(word in step_text_lower for word in ["review", "document", "report"]):
                    categories[step_key] = "Documentation"
                else:
                    categories[step_key] = "General"
            
            result.categories = categories
            
        except Exception as e:
            self.logger.warning(f"Category analysis failed: {e}")
            result.categories = {}
        
        return result
    
    def _calculate_complexity_score(self, request: str, result: PlanningResult) -> float:
        """Calculate complexity score for the request."""
        try:
            score = 0.0
            
            # Base score from number of steps
            score += len(result.steps) * 0.1
            
            # Add score based on request length
            score += min(len(request.split()), 50) * 0.01
            
            # Add score for complexity keywords
            complexity_keywords = [
                "complex", "advanced", "sophisticated", "comprehensive", "detailed",
                "multiple", "various", "different", "integrate", "coordinate",
                "optimize", "algorithm", "system", "architecture", "framework"
            ]
            
            request_lower = request.lower()
            for keyword in complexity_keywords:
                if keyword in request_lower:
                    score += 0.05
            
            # Normalize to 0-1 range
            score = min(score, 1.0)
            
            return round(score, 3)
            
        except Exception:
            return 0.5
    
    def _calculate_confidence_score(self, request: str, plan_text: str, steps: Dict[str, str]) -> float:
        """Calculate confidence score for the generated plan."""
        try:
            if not steps or not plan_text:
                return 0.0
            
            # Base score from number of valid steps
            base_score = min(len(steps) / 5.0, 1.0) * 0.4
            
            # Score from step text quality (average length)
            avg_step_length = sum(len(step.split()) for step in steps.values()) / len(steps)
            length_score = min(avg_step_length / 10.0, 1.0) * 0.3
            
            # Score from text coherence (simple heuristic)
            coherence_score = 0.3
            if "step" in plan_text.lower():
                coherence_score = 0.3
            
            total_score = base_score + length_score + coherence_score
            return round(total_score, 3)
            
        except Exception:
            return 0.5
    
    def _merge_configs(self, custom_config: Dict[str, Any]) -> PlanningConfig:
        """Merge custom configuration with default configuration."""
        config_dict = self.config.__dict__.copy()
        config_dict.update(custom_config)
        
        # Create new config object
        merged_config = PlanningConfig()
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
            "max_steps": self.config.max_steps,
            "use_chat_template": self.config.use_chat_template
        }


# Factory function for easy instantiation
def create_planner(
    model_name: str = "google-t5/t5-base",
    strategy: str = "sequential",
    max_steps: int = 10,
    device: str = "auto"
) -> LLMPlanner:
    """
    Factory function to create a planner with common settings.
    
    Args:
        model_name: HuggingFace model name
        strategy: Planning strategy ('sequential', 'hierarchical', 'dependency_based', 'category_based', 'priority_based')
        max_steps: Maximum number of steps to generate
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Configured LLMPlanner instance
    """
    
    strategy_enum = PlanningStrategy(strategy)
    
    config = PlanningConfig(
        model_name=model_name,
        strategy=strategy_enum,
        max_steps=max_steps,
        device=device
    )
    
    return LLMPlanner(config)


# CLI Interface and Testing
if __name__ == "__main__":
    import argparse
    
    def main():
        """Main function for CLI usage."""
        parser = argparse.ArgumentParser(description="LLM Request Planning and Decomposition")
        
        parser.add_argument("--request", "-r", required=True, help="Complex request to break down")
        parser.add_argument("--model", "-m", default="google-t5/t5-base", help="Model to use")
        parser.add_argument("--strategy", "-s", choices=["sequential", "hierarchical", "dependency_based", "category_based", "priority_based"], 
                          default="sequential", help="Planning strategy")
        parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of steps")
        parser.add_argument("--output", "-o", help="Output file (JSON format)")
        parser.add_argument("--dependencies", action="store_true", help="Analyze step dependencies")
        parser.add_argument("--priorities", action="store_true", help="Analyze step priorities")
        parser.add_argument("--categories", action="store_true", help="Analyze step categories")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        
        args = parser.parse_args()
        
        # Setup logging
        if args.verbose:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        
        try:
            # Create config
            config = PlanningConfig(
                model_name=args.model,
                strategy=PlanningStrategy(args.strategy),
                max_steps=args.max_steps,
                include_dependencies=args.dependencies,
                include_priorities=args.priorities,
                include_categories=args.categories
            )
            
            # Create planner
            print("🚀 Initializing LLM Planner...")
            planner = LLMPlanner(config)
            
            # Process request
            print("📝 Processing request...")
            result = planner.plan_request(args.request)
            
            # Display results
            print("\n" + "="*60)
            print("📋 REQUEST PLANNING RESULTS")
            print("="*60)
            print(f"🎯 Original Request: {result.original_request}")
            print(f"📊 Strategy Used: {result.strategy_used}")
            print(f"🔢 Total Steps: {result.total_steps}")
            print(f"⏱️  Processing Time: {result.processing_time:.2f} seconds")
            print(f"🎯 Confidence Score: {result.confidence_score:.3f}")
            print(f"🤖 Model Used: {result.model_used}")
            
            if result.complexity_score > 0:
                print(f"📈 Complexity Score: {result.complexity_score:.3f}")
            
            print(f"\n📝 GENERATED PLAN:")
            print("-" * 40)
            print(result.get_formatted_plan())
            
            print(f"\n🔧 JSON OUTPUT:")
            print("-" * 40)
            print(result.to_json())
            
            # Save to file if requested
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result.to_json())
                print(f"\n💾 Results saved to: {args.output}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
        
        return 0
    
    # Example usage and testing
    def test_basic_functionality():
        """Test basic functionality with sample requests."""
        
        sample_requests = [
            "Create a web application for managing a small library with user authentication, book catalog, and borrowing system",
            "Analyze the performance issues in our database and implement optimization strategies",
            "Design and implement a machine learning model to predict customer churn for our subscription service"
        ]
        
        print("🧪 Testing LLM Planner...")
        
        try:
            # Test with different strategies
            strategies = ["sequential", "hierarchical", "dependency_based"]
            
            for i, request in enumerate(sample_requests, 1):
                print(f"\n--- Test {i}: {strategies[(i-1) % len(strategies)]} strategy ---")
                print(f"Request: {request[:80]}...")
                
                planner = create_planner(
                    model_name="google-t5/t5-small",  # Use smaller model for testing
                    strategy=strategies[(i-1) % len(strategies)],
                    max_steps=5
                )
                
                result = planner.plan_request(request)
                
                print(f"Steps generated: {result.total_steps}")
                print(f"Strategy: {result.strategy_used}")
                print(f"Confidence: {result.confidence_score:.3f}")
                print(f"Processing time: {result.processing_time:.2f}s")
                
                # Show first 2 steps
                for step_key in sorted(list(result.steps.keys())[:2]):
                    step_num = step_key.replace('step', '')
                    print(f"  {step_num}. {result.steps[step_key][:60]}...")
            
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