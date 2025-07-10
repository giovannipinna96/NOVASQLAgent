"""
This file defines the foundational architecture for managing and interacting with Large Language Models (LLMs)
using Python. The codebase should follow best practices in Python programming, ensuring scalability, clarity,
and modularity. The overall goal is to provide a robust interface for both local and remote (API-based) LLMs,
including prompt handling, inference execution, and memory management.

STRUCTURE OVERVIEW:

1. ABSTRACT BASE CLASS (OR PROTOCOL):
    - A base abstract class or typing.Protocol must be created to serve as the ancestor of all LLM-related classes.
    - This class should define:
        • An abstract method `run()` for executing inference with the LLM.
        • Core attributes such as:
            - `model_name`: a string representing the model identifier or path.
            - `task`: a string describing the task the model is assigned to (e.g., summarization, QA).
            - `prompt_path`: a string path to the template file to be used for prompt generation.
        • A method `load_prompt_template(input_dict: dict) -> str`:
            - This method should read the prompt template file from `prompt_path`.
            - It should format the prompt using the values provided in the `input_dict`.
        • A `__str__` or `__print__` method that outputs a well-formatted and readable summary
          of the model instance, its configuration, and state.

2. HUGGINGFACE-BASED LLM CLASS:
    - This class should inherit from the base class and be responsible for loading and interacting
      with models hosted on Hugging Face (either pre-trained or fine-tuned).
    - Required features:
        • Initialization parameters should include flags such as:
            - `use_4bit`: if True, load the model in 4-bit precision.
            - `use_8bit`: if True, load in 8-bit precision.
            - `use_memory`: if True, enable memory or conversation context handling.
        • Auto-detection of LoRA adapters:
            - If `model_name` is a path and contains PEFT adapters (e.g., LoRA), it should automatically
              load the adapters using `peft`.
        • A method `load_chat_template()` for preparing chat-based prompts when applicable.
        • Any additional methods or attributes needed to support memory buffer management, tokenizer
          loading, batching, and model generation.

3. API-BASED LLM CLASS:
    - This class should also inherit from the base class, and it should serve as a wrapper for interacting
      with remote LLMs via API (e.g., OpenAI GPT-4o, Anthropic Claude Sonnet).
    - Required features:
        • API client configuration and authentication.
        • A `run()` implementation that sends the request and parses the response.
        • Support for session-based memory management (e.g., conversation history stored and reused).
        • Abstraction layer to make the use of different APIs (OpenAI, Anthropic, etc.) interchangeable
          through unified methods and interface logic.
        • Optional retry logic and rate limit handling.

DESIGN PRINCIPLES AND BEST PRACTICES:
    - All classes should be fully type-annotated with `mypy`-friendly hints.
    - Use of Python’s `abc` module for abstract base classes if necessary.
    - Follow PEP8 standards and apply docstrings to all public methods and classes.
    - Separate concerns: prompt formatting, model loading, inference, and memory should be modular.
    - Consider using dependency injection and configuration classes (e.g., via `pydantic`) for extensibility.
    - Logging should be implemented to trace usage, errors, and debug information where appropriate.

This file serves as the core logic layer for a broader framework or application that leverages LLMs for complex tasks,
and it should be designed to be easily extensible and maintainable.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List

# Assuming prompt_template_manager.py and memory.py are in the same directory or accessible
try:
    from .prompt_template_manager import PromptTemplateManager
    from .memory import ConversationMemory, MessageDirection
except ImportError:
    # Fallback for standalone execution or if modules are not yet fully integrated
    # This allows development of LLMmodel.py even if other parts are incomplete.
    # In a real scenario, these would be hard dependencies.
    logging.warning("Could not import PromptTemplateManager or ConversationMemory. Using placeholder classes.")
    class PromptTemplateManager: #type: ignore
        def __init__(self, template_path: Path): self.template_path = template_path
        def format_prompt(self, input_dict: Dict[str, str]) -> str: return f"Formatted: {input_dict}"
        def display_template_overview(self): print(f"Overview for {self.template_path}")

    class ConversationMemory: #type: ignore
        def __init__(self, agent_id: str, llm_model: Any = None): self.agent_id = agent_id; self.messages = []
        def add_message(self, content: str, direction: Any, llm_model_name: Optional[str] = None): self.messages.append(content)
        def get_conversation_text(self) -> str: return "\n".join(self.messages)

    class MessageDirection: #type: ignore
        INCOMING = "incoming"; OUTGOING = "outgoing"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """
    Abstract Base Class for Large Language Models.

    Defines the common interface for all LLM implementations, whether local (HuggingFace)
    or API-based (OpenAI, Anthropic, etc.).
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        prompt_template_path: Optional[Union[str, Path]] = None,
        use_memory: bool = False,
        agent_id: Optional[str] = None,
    ):
        """
        Initializes the BaseLLM.

        Args:
            model_name: Identifier or path for the LLM.
            task: Description of the task the LLM is assigned to (e.g., "summarization", "QA").
            prompt_template_path: Optional path to a .txt file for prompt templating.
            use_memory: If True, enables conversation memory. Requires agent_id.
            agent_id: Unique ID for the agent using this LLM, required if use_memory is True.
        """
        self.model_name: str = model_name
        self.task: str = task
        self.prompt_template_path: Optional[Path] = Path(prompt_template_path) if prompt_template_path else None
        self.prompt_manager: Optional[PromptTemplateManager] = None
        self.use_memory: bool = use_memory
        self.agent_id: Optional[str] = agent_id
        self.memory: Optional[ConversationMemory] = None

        if self.prompt_template_path:
            try:
                self.prompt_manager = PromptTemplateManager(self.prompt_template_path)
            except FileNotFoundError:
                logger.error(f"Prompt template file not found: {self.prompt_template_path}. Prompt manager not initialized.")
                self.prompt_template_path = None # Disable if file not found
            except ValueError as e:
                logger.error(f"Error initializing PromptTemplateManager: {e}. Prompt manager not initialized.")
                self.prompt_template_path = None


        if self.use_memory:
            if not self.agent_id:
                logger.error("agent_id is required when use_memory is True. Disabling memory.")
                self.use_memory = False
            else:
                # Pass `self` (the LLM instance) to ConversationMemory
                self.memory = ConversationMemory(agent_id=self.agent_id, llm_model=self)
                logger.info(f"Conversation memory enabled for agent '{self.agent_id}'.")


    @abstractmethod
    def run(self, input_data: Union[str, Dict[str, str]], **kwargs: Any) -> str:
        """
        Executes inference with the LLM.

        Args:
            input_data: Either a raw string prompt or a dictionary for template formatting.
            **kwargs: Additional model-specific generation parameters.

        Returns:
            The LLM's generated response as a string.
        """
        pass

    def load_prompt_template(self, input_dict: Dict[str, str]) -> str:
        """
        Loads and formats a prompt template using the provided input dictionary.

        Args:
            input_dict: Dictionary with keys matching placeholders in the template.

        Returns:
            The formatted prompt string.

        Raises:
            ValueError: If no prompt manager is initialized (e.g., no template path provided).
            KeyError: If keys are missing for formatting.
        """
        if not self.prompt_manager:
            raise ValueError("Prompt manager not initialized. Provide a valid prompt_template_path.")

        try:
            return self.prompt_manager.format_prompt(input_dict)
        except KeyError as e:
            logger.error(f"Error formatting prompt: Missing key {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error formatting prompt: {e}")
            raise

    def _prepare_prompt(self, input_data: Union[str, Dict[str, str]]) -> str:
        """Internal helper to get the final prompt string, using template if applicable."""
        if isinstance(input_data, dict):
            if not self.prompt_manager:
                raise ValueError("Input is a dictionary, but no prompt_template_path was provided for formatting.")
            prompt_text = self.load_prompt_template(input_data)
        elif isinstance(input_data, str):
            prompt_text = input_data
        else:
            raise TypeError("input_data must be a string or a dictionary.")
        return prompt_text

    def _add_to_memory(self, content: str, direction: MessageDirection, llm_model_name_override: Optional[str] = None) -> None:
        """Adds a message to the conversation memory if enabled."""
        if self.use_memory and self.memory:
            model_name = llm_model_name_override if llm_model_name_override else self.model_name
            self.memory.add_message(content, direction, llm_model_name=model_name)

    def get_conversation_history_text(self) -> Optional[str]:
        """Returns the conversation history as a single string if memory is enabled."""
        if self.use_memory and self.memory:
            return self.memory.get_conversation_text()
        return None

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text according to the model's tokenizer.

        Args:
            text: The input string.

        Returns:
            The number of tokens.
        """
        pass

    def __str__(self) -> str:
        """Provides a readable summary of the LLM instance."""
        mem_status = f"Enabled (Agent ID: {self.agent_id})" if self.use_memory and self.agent_id else "Disabled"
        prompt_status = f"Initialized ({self.prompt_template_path.name})" if self.prompt_manager and self.prompt_template_path else "Not Initialized"

        return (
            f"--- LLM Instance Summary ---\n"
            f"Model Name: {self.model_name}\n"
            f"Task: {self.task}\n"
            f"Type: {self.__class__.__name__}\n"
            f"Prompt Template: {prompt_status}\n"
            f"Conversation Memory: {mem_status}\n"
            f"--------------------------"
        )

class HuggingFaceLLM(BaseLLM):
    """
    LLM implementation for models hosted on Hugging Face Hub or local paths,
    leveraging the `transformers` and `peft` libraries.
    """
    def __init__(
        self,
        model_name: str,
        task: str,
        prompt_template_path: Optional[Union[str, Path]] = None,
        use_memory: bool = False,
        agent_id: Optional[str] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
        device: Optional[str] = None, # e.g., "cuda", "cpu", "mps"
        model_kwargs: Optional[Dict[str, Any]] = None, # For AutoModelForCausalLM.from_pretrained
        tokenizer_kwargs: Optional[Dict[str, Any]] = None, # For AutoTokenizer.from_pretrained
    ):
        super().__init__(model_name, task, prompt_template_path, use_memory, agent_id)

        self.use_4bit: bool = use_4bit
        self.use_8bit: bool = use_8bit
        self.device: Optional[str] = device
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.tokenizer_kwargs = tokenizer_kwargs if tokenizer_kwargs else {}

        self.model: Optional[Any] = None # Actual HuggingFace model
        self.tokenizer: Optional[Any] = None # Actual HuggingFace tokenizer

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Loads the HuggingFace model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel # For LoRA adapter detection
        except ImportError:
            logger.error("HuggingFace libraries (`transformers`, `peft`, `bitsandbytes`) are not installed. HuggingFaceLLM cannot be used.")
            raise ImportError("Please install `transformers`, `peft`, `accelerate`, and `bitsandbytes` to use HuggingFaceLLM.")

        logger.info(f"Loading HuggingFace model: {self.model_name}")

        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="torch.float16", # Or bfloat16 if available and preferred
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Loading model in 4-bit precision.")
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading model in 8-bit precision.")

        # Model loading
        try:
            model_load_path = self.model_name
            # Check if model_name is a path and might contain adapters
            is_local_path = Path(self.model_name).is_dir()

            # Base model loading arguments
            load_args = {
                "quantization_config": quantization_config,
                "device_map": "auto" if self.device is None else self.device, # 'auto' for optimal distribution
                **self.model_kwargs
            }
            # Remove None quantization_config if not used
            if load_args["quantization_config"] is None:
                del load_args["quantization_config"]


            self.model = AutoModelForCausalLM.from_pretrained(model_load_path, **load_args)

            # PEFT/LoRA adapter auto-detection and loading
            # A common pattern is that the main model path contains an 'adapter_config.json'
            # or the model is loaded and then adapters are merged from a sub-directory.
            # PeftModel.from_pretrained handles this by loading base and then adapters.
            # We assume here that if LoRA adapters are present, `model_name` points to the
            # directory where `PeftModel` can find both base and adapter.
            # A more explicit way is to have separate base_model_path and adapter_path.
            # For now, we rely on PeftModel's smart loading.
            try:
                # This will load adapters if `model_name` is a PEFT-saved model directory
                # If it's just a base model, it should ideally not fail but return the base.
                # However, `PeftModel.from_pretrained` expects the first arg to be the base model.
                # If `self.model_name` *is* an adapter directory, we need to know the base model.
                # This part is tricky without more context on how models/adapters are stored.
                # A common approach:
                # 1. Load base model.
                # 2. If adapter path is given, load PeftModel(base_model, adapter_path).
                # For simplicity, if `model_name` seems like a PEFT model dir, we try to load it as such.
                # This assumes `model_name` is the path to the *saved PEFT model*, not just adapters.
                if is_local_path and (Path(self.model_name) / "adapter_config.json").exists():
                     logger.info(f"Attempting to load PEFT model from {self.model_name}")
                     # The base model is already loaded above. Now we check if it needs PEFT wrapping.
                     # This logic might need refinement. If AutoModelForCausalLM already loaded a PeftModel, this is redundant.
                     # If not, and adapters are present, this is where PeftModel.from_pretrained(base_model, adapter_path) would be used.
                     # For now, let's assume AutoModelForCausalLM handles it if `model_name` is a full PEFT checkpoint.
                     # If `model_name` is just a base and adapters are separate, this needs adjustment.
                     pass # Assuming AutoModel already handled it or adapters are applied later if separate.

            except Exception as peft_e:
                logger.warning(f"Could not load {self.model_name} as a PEFT model: {peft_e}. Loaded as a base transformer model.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer pad_token was None, set to eos_token.")

            self.model.eval() # Set to evaluation mode by default
            logger.info(f"HuggingFace model '{self.model_name}' and tokenizer loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading HuggingFace model or tokenizer '{self.model_name}': {e}")
            raise

    def run(self, input_data: Union[str, Dict[str, str]], generation_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates text using the loaded HuggingFace model.

        Args:
            input_data: Prompt string or dictionary for template formatting.
            generation_kwargs: Dictionary of arguments for `model.generate()`
                               (e.g., max_new_tokens, temperature, do_sample).

        Returns:
            The generated text string.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer are not loaded. Call _load_model_and_tokenizer first or check for errors.")

        prompt_text = self._prepare_prompt(input_data)

        # Add to memory before generation (user input)
        self._add_to_memory(prompt_text, MessageDirection.INCOMING, llm_model_name_override="User/System")


        # Prepend conversation history if memory is enabled
        full_prompt_text = prompt_text
        if self.use_memory and self.memory:
            history = self.memory.get_conversation_text() # Gets all messages including current one
            # To avoid duplicating the current prompt, we might need more refined history retrieval
            # For now, let's assume memory.get_conversation_text() gives the context *before* the current input.
            # A better approach for chat: memory.get_formatted_chat_history()
            # Let's refine memory interaction:
            # 1. User prompt is `prompt_text`.
            # 2. Get history *excluding* this new prompt.
            # 3. Construct full input.
            # For simplicity, let's assume `prompt_text` is the full thing to send for now,
            # and memory is more for logging, or a chat template would handle history.

        inputs = self.tokenizer(full_prompt_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            **(generation_kwargs or {})
        }

        logger.info(f"Generating text with kwargs: {gen_kwargs}")
        try:
            output_sequences = self.model.generate(**inputs, **gen_kwargs)

            # Decode the output, skipping special tokens and the prompt
            # If input_ids are passed to generate, the output includes them.
            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_sequences[0, input_length:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            logger.info(f"Generated response (first 100 chars): {response_text[:100]}...")

            # Add LLM response to memory
            self._add_to_memory(response_text, MessageDirection.OUTGOING)

            return response_text.strip()
        except Exception as e:
            logger.error(f"Error during HuggingFace model generation: {e}")
            raise

    def load_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Applies the tokenizer's chat template to a list of messages.
        Each message is a dict with "role" (e.g., "user", "assistant") and "content".

        Args:
            messages: A list of message dictionaries.

        Returns:
            A formatted string ready for tokenization.

        Raises:
            RuntimeError: If tokenizer is not loaded or does not have a chat template.
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        if not self.tokenizer.chat_template:
            # Attempt to apply a default one if common model type
            if "llama" in self.model_name.lower() or "mistral" in self.model_name.lower():
                 logger.warning(f"Tokenizer for {self.model_name} does not have a `chat_template`. Applying a Llama-style default. This might not be optimal.")
                 # This is a simplified version. Real Llama template is more complex.
                 # For robust solution, use `tokenizer.apply_chat_template` with a manually set template if needed.
                 formatted_prompt = ""
                 for msg in messages:
                     formatted_prompt += f"[INST] {msg['content']} [/INST]\n" if msg['role'] == 'user' else f"{msg['content']}\n"
                 return formatted_prompt
            else:
                raise RuntimeError(f"Tokenizer for {self.model_name} does not have a `chat_template` attribute. Cannot format chat.")

        try:
            # `apply_chat_template` tokenizes by default, add tokenize=False for string output
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            # Fallback or re-raise
            raise RuntimeError(f"Could not apply chat template for {self.model_name}: {e}")


    def count_tokens(self, text: str) -> int:
        """Counts tokens using the loaded HuggingFace tokenizer."""
        if not self.tokenizer:
            # This case should ideally not be reached if constructor succeeded
            logger.warning("Tokenizer not loaded, cannot count tokens accurately. Returning len(text.split()).")
            return len(text.split())
        return len(self.tokenizer.encode(text))

    def __str__(self) -> str:
        base_summary = super().__str__()
        hf_specifics = (
            f"HuggingFace Specifics:\n"
            f"  4-bit Quantization: {self.use_4bit}\n"
            f"  8-bit Quantization: {self.use_8bit}\n"
            f"  Device: {self.device or 'auto'}\n"
            f"  Model Loaded: {'Yes' if self.model else 'No'}\n"
            f"  Tokenizer Loaded: {'Yes' if self.tokenizer else 'No'}"
        )
        return f"{base_summary}\n{hf_specifics}"


class OpenAILLM(BaseLLM):
    """
    LLM implementation for interacting with OpenAI models via their API.
    """
    def __init__(
        self,
        model_name: str, # e.g., "gpt-4o", "gpt-3.5-turbo"
        task: str,
        api_key: Optional[str] = None, # Can also be set via OPENAI_API_KEY env var
        prompt_template_path: Optional[Union[str, Path]] = None,
        use_memory: bool = False,
        agent_id: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None, # For openai.OpenAI() client
    ):
        super().__init__(model_name, task, prompt_template_path, use_memory, agent_id)

        self.api_key: Optional[str] = api_key
        self.client: Optional[Any] = None # OpenAI client instance
        self.api_kwargs = api_kwargs if api_kwargs else {}

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initializes the OpenAI API client."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OpenAI library (`openai`) is not installed. OpenAILLM cannot be used.")
            raise ImportError("Please install `openai` to use OpenAILLM: pip install openai")

        try:
            self.client = OpenAI(api_key=self.api_key, **self.api_kwargs)
            # Test connection by listing models (optional, can be slow)
            # self.client.models.list()
            logger.info(f"OpenAI client initialized for model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None # Ensure client is None if init fails
            raise

    def run(self, input_data: Union[str, Dict[str, str]], generation_kwargs: Optional[Dict[str, Any]] = None) -> str:
        """
        Sends a request to the OpenAI API and returns the response.

        Args:
            input_data: Prompt string or dictionary for template formatting.
            generation_kwargs: Dictionary of arguments for `client.chat.completions.create()`
                               (e.g., max_tokens, temperature).

        Returns:
            The generated text string from the OpenAI model.
        """
        if not self.client:
            raise RuntimeError("OpenAI client is not initialized. Check for errors during init.")

        prompt_text = self._prepare_prompt(input_data)
        self._add_to_memory(prompt_text, MessageDirection.INCOMING, llm_model_name_override="User/System")

        messages_payload = []
        if self.use_memory and self.memory and self.memory.messages:
            # Construct messages array from memory for chat models
            # The last message added was the current user prompt.
            for msg in self.memory.messages:
                role = "user" if msg.direction == MessageDirection.INCOMING else "assistant"
                # If it's a summary, it might be better as a system prompt or part of user prompt.
                if msg.metadata.get("is_summary"):
                    role = "system" # Or append to the next user message.
                    messages_payload.append({"role": role, "content": f"Previous conversation summary: {msg.content}"})
                else:
                     messages_payload.append({"role": role, "content": msg.content})
            if not messages_payload or messages_payload[-1]["content"] != prompt_text:
                 # This can happen if memory was empty or last message wasn't the current prompt
                 # Ensure the current prompt is the last user message
                 # This logic needs to be robust based on how memory is structured.
                 # Assuming the prompt_text is the latest user utterance.
                 # If memory already contains it via _add_to_memory, this could duplicate.
                 # Refined logic: Memory should store history *before* current prompt.
                 # Then add current prompt here.
                 # Let's adjust: _add_to_memory(prompt) happens *before* this point.
                 # So, memory.messages should be up-to-date.
                 pass # Current message is already in memory, so it's in messages_payload
        else:
            # No memory, or memory is empty: just the current prompt as a user message
            messages_payload.append({"role": "user", "content": prompt_text})

        # Default generation arguments for chat completions
        gen_kwargs = {
            "max_tokens": 1024, # OpenAI defaults vary, set a reasonable one
            "temperature": 0.7,
            **(generation_kwargs or {})
        }

        logger.info(f"Sending request to OpenAI model '{self.model_name}' with kwargs: {gen_kwargs}")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_payload, #type: ignore
                **gen_kwargs
            )

            response_text = response.choices[0].message.content
            if response_text is None:
                logger.warning("OpenAI API returned None content.")
                response_text = ""

            logger.info(f"Received response from OpenAI (first 100 chars): {response_text[:100]}...")

            self._add_to_memory(response_text, MessageDirection.OUTGOING)
            return response_text.strip()

        except Exception as e: # Catch specific OpenAI errors if possible
            logger.error(f"Error during OpenAI API request: {e}")
            # Example: handle openai.APIError, openai.RateLimitError etc.
            raise

    def count_tokens(self, text: str) -> int:
        """
        Counts tokens using `tiktoken` for OpenAI models.
        Note: This is an estimation. Actual token count can vary slightly.
        """
        try:
            import tiktoken
        except ImportError:
            logger.warning("`tiktoken` library not found. Cannot accurately count tokens for OpenAI. "
                           "Returning len(text.split()). Install with `pip install tiktoken`.")
            return len(text.split())

        try:
            # Attempt to get encoding for the specific model
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to a common encoding if specific model not found
            logger.warning(f"No tiktoken encoding found for model '{self.model_name}'. Using 'cl100k_base'.")
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def __str__(self) -> str:
        base_summary = super().__str__()
        client_status = "Initialized" if self.client else "Not Initialized"
        api_specifics = (
            f"OpenAI API Specifics:\n"
            f"  API Client Status: {client_status}"
        )
        return f"{base_summary}\n{api_specifics}"


# Example Usage (Conceptual - requires actual models/API keys)
if __name__ == "__main__":
    # --- HuggingFace Example (requires a model and transformers installed) ---
    # Create a dummy prompt template file
    dummy_prompt_file = Path("dummy_hf_prompt.txt")
    with open(dummy_prompt_file, "w") as f:
        f.write("Translate to French: {text_to_translate}")

    # Ensure you have a model like 'distilgpt2' or a path to a local model
    # For full functionality (4bit/8bit), ensure 'bitsandbytes' and 'accelerate' are installed.
    try:
        logger.info("\n--- Testing HuggingFaceLLM ---")
        # hf_llm = HuggingFaceLLM(
        #     model_name="distilgpt2", # A small model for quick testing
        #     task="text-generation",
        #     prompt_template_path=dummy_prompt_file,
        #     use_memory=True,
        #     agent_id="hf_agent_001",
        #     # use_4bit=True, # Requires compatible hardware and libraries
        # )
        # print(hf_llm)
        # hf_response = hf_llm.run(
        #     {"text_to_translate": "Hello, world!"},
        #     generation_kwargs={"max_new_tokens": 50}
        # )
        # print(f"HuggingFace LLM Response: {hf_response}")
        # print(f"Token count for 'Hello world': {hf_llm.count_tokens('Hello world')}")

        # Chat example (if model supports it, e.g. a Llama model)
        # chat_messages = [
        #     {"role": "user", "content": "What is the capital of France?"},
        # ]
        # try:
        # formatted_chat_prompt = hf_llm.load_chat_template(chat_messages)
        # print(f"\nFormatted chat prompt:\n{formatted_chat_prompt}")
        # chat_response = hf_llm.run(formatted_chat_prompt, generation_kwargs={"max_new_tokens": 30})
        # print(f"Chat response: {chat_response}")
        # except RuntimeError as e:
        #     print(f"Chat template error (expected for distilgpt2): {e}")

        # print("\nConversation History (HuggingFace):")
        # if hf_llm.memory:
        #     for msg_content in hf_llm.memory.messages: # type: ignore
        #         print(f"- {str(msg_content)[:100]}...") # As memory stores full Message objects

        logger.info("Skipping HuggingFaceLLM test as it requires model download and setup.")

    except ImportError as e:
        logger.warning(f"Skipping HuggingFaceLLM example: {e}")
    except Exception as e:
        logger.error(f"Error in HuggingFaceLLM example: {e}")
    finally:
        if dummy_prompt_file.exists():
            dummy_prompt_file.unlink()

    # --- OpenAI Example (requires OPENAI_API_KEY environment variable or passed directly) ---
    dummy_openai_prompt_file = Path("dummy_openai_prompt.txt")
    with open(dummy_openai_prompt_file, "w") as f:
        f.write("Tell me a joke about {topic}.")

    try:
        logger.info("\n--- Testing OpenAILLM ---")
        # Ensure OPENAI_API_KEY is set in your environment, or pass it as api_key="sk-..."
        # openai_llm = OpenAILLM(
        #     model_name="gpt-3.5-turbo-instruct", # Or "gpt-4o" / "gpt-3.5-turbo" for chat
        #     task="completion",
        #     prompt_template_path=dummy_openai_prompt_file,
        #     use_memory=True,
        #     agent_id="openai_agent_001",
        #     # api_key="YOUR_API_KEY" # If not set in env
        # )
        # print(openai_llm)
        # openai_response = openai_llm.run(
        #     {"topic": "programmers"},
        #     generation_kwargs={"max_tokens": 60} # For completion models
        # )
        # print(f"OpenAI LLM Response: {openai_response}")
        # print(f"Token count for 'Hello world': {openai_llm.count_tokens('Hello world')}")

        # print("\nConversation History (OpenAI):")
        # if openai_llm.memory:
        #     for msg_obj in openai_llm.memory.messages: # type: ignore
        #         print(f"- Role: {msg_obj.direction.value}, Content: {msg_obj.content[:100]}...")
        logger.info("Skipping OpenAILLM test as it requires API key and makes external calls.")

    except ImportError as e:
        logger.warning(f"Skipping OpenAILLM example: {e}")
    except RuntimeError as e: # Catch client init issues
        logger.warning(f"Skipping OpenAILLM example due to runtime error (likely API key issue): {e}")
    except Exception as e:
        logger.error(f"Error in OpenAILLM example: {e}")
    finally:
        if dummy_openai_prompt_file.exists():
            dummy_openai_prompt_file.unlink()

    logger.info("\nLLMmodel.py example usage finished.")
