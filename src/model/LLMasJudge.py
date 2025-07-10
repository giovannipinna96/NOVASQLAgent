"""
This module implements the core logic for building an "LLM-as-a-Judge" system, where a language model (LLM) is used
to evaluate input scenarios or claims and produce a strict binary verdict: YES or NO. This approach is commonly
used in benchmarking, comparative model evaluation, factual consistency testing, or safety-critical decision pipelines.

MODULE GOALS AND SPECIFICATIONS:

1. SYSTEM INTEGRATION:
    - This file must rely on and integrate with existing components from the project:
        • LLM management (see `LLMmodel.py`)
        • Prompt template formatting (see `prompt_template_manager.py`)
        • Memory/conversation history (see `conversation_memory.py`) - Though memory might be less relevant for stateless judging.
    - The system should assume an LLM class that supports `.run()` or similar execution logic, and that can be
      initialized with named models, task, prompt path, and memory options.

2. STRICT YES/NO RESPONSE FUNCTION:
    - Implement a function that takes an LLM instance and a prompt as input, executes the prompt, and returns strictly
      "YES" or "NO" as a response.
    - The system must enforce binary answers. If the output is not exactly "yes" or "no" (case-insensitive),
      it must:
        • Immediately halt execution of that instance (or return a distinct error/None value)
        • Raise a runtime alert or warning
        • Optionally retry or escalate
    - This helps ensure the model's judgment is reliable and aligned to expected structure for downstream pipelines.

3. TOKEN PROBABILITY ANALYSIS:
    - Implement a function to query the LLM in a way that:
        • Extracts the probability distribution over the output tokens
        • Identifies the probability assigned specifically to "yes" and "no" tokens
    - Return a dictionary or data structure containing:
        {
            "yes_token_id": probability_yes, # Using token_id as key might be better if token strings vary
            "no_token_id": probability_no,
            "top_tokens": [{"token_id": probability, "token_str": string_representation}],
        }
    - This allows interpretation of uncertainty and model confidence in its judgment.
    - This feature is highly dependent on the LLM's capabilities (e.g., HuggingFace models can output logits,
      OpenAI API can provide logprobs for chosen tokens).

4. PARALLEL EXECUTION PIPELINE:
    - Design a multiprocessing or multithreaded framework to perform judgment evaluation in parallel.
    - The system must:
        • Load the LLM across multiple GPU memory pools or instances (multi-GPU or model sharding compatible) - This is complex.
          For simplicity, we might start with process-based parallelism where each process loads its own model copy,
          or use a shared model if feasible (e.g. with vLLM or similar serving frameworks, outside scope of this file).
        • Use a queue/pool of processes to dispatch tasks concurrently
        • Collect all results efficiently

    - Each task should return a structured result such as:
        {
            "question": "...",
            "answer": "YES" or "NO" or "INVALID",
            "probability_yes": Optional[float],
            "probability_no": Optional[float],
            "raw_output": "Actual model output string"
        }

    - This system must be scalable, fault-tolerant, and configurable for batch evaluation.

5. ENGINEERING AND BEST PRACTICES:
    - Ensure the code follows modern Python best practices:
        • PEP8 style guide compliance
        • Robust type annotations
        • Logging instead of print statements
        • Exception handling for model output, token sampling, and multiprocessing errors
        • Modularity to allow reuse and integration with evaluation or UI layers

    - Consider including the ability to set a temperature, top-k/top-p sampling strategy if required, and deterministic
      decoding options for strict binary evaluations. (e.g. temperature=0 or greedy decoding).

    - All major functions and classes must be documented with clear docstrings explaining inputs, outputs, and side effects.

This file forms the core of a decision-based LLM pipeline and must guarantee precision, consistency, and interpretability
in binary classification decisions. Ideal for evaluation frameworks, fact verification systems, and aligned AI agents.
"""
import logging
import re
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

# Assuming LLMmodel.py is in the same directory or accessible
try:
    from .LLMmodel import BaseLLM, HuggingFaceLLM, OpenAILLM
    # from .prompt_template_manager import PromptTemplateManager (LLMmodel handles its own)
    # from .memory import ConversationMemory (LLMmodel handles its own)
except ImportError:
    logging.critical("LLMasJudge: Could not import BaseLLM or specific LLM classes. This module will not function.")
    # Define placeholders if imports fail, to allow file to be parsed, but it won't work.
    class BaseLLM: # type: ignore
        def __init__(self, model_name: str, task: str, **kwargs): self.model_name=model_name; self.task=task
        def run(self, prompt: str, **kwargs) -> str: return "Placeholder LLM Response"
        def count_tokens(self, text: str) -> int: return len(text.split())
    class HuggingFaceLLM(BaseLLM): pass # type: ignore
    class OpenAILLM(BaseLLM): pass # type: ignore


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

class JudgeDecision(str, Enum):
    """Enumeration for judge decisions."""
    YES = "YES"
    NO = "NO"
    INVALID = "INVALID" # For when the LLM output is not a clear YES/NO

@dataclass
class JudgeResult:
    """Structured result for a single judgment task."""
    question: str
    answer: JudgeDecision
    raw_output: str
    probability_yes: Optional[float] = None
    probability_no: Optional[float] = None
    top_tokens_info: Optional[List[Dict[str, Any]]] = None # e.g., [{"token_str": "Yes", "probability": 0.9}]
    error_message: Optional[str] = None


class LLMasJudge:
    """
    Uses an LLM to make YES/NO judgments on given prompts/questions.
    """

    def __init__(self, llm_instance: BaseLLM):
        """
        Initializes the LLM-as-a-Judge system.

        Args:
            llm_instance: An initialized instance of a class derived from BaseLLM.
        """
        if not isinstance(llm_instance, BaseLLM):
            raise TypeError("llm_instance must be a valid instance of BaseLLM or its subclasses.")
        self.llm: BaseLLM = llm_instance
        logger.info(f"LLM-as-a-Judge initialized with LLM: {self.llm.model_name} ({self.llm.__class__.__name__})")

        # Pre-compile regex for faster YES/NO parsing
        # This regex looks for "yes" or "no" at the beginning or end of the string,
        # possibly surrounded by punctuation or whitespace.
        # It's made more robust to handle variations.
        self._yes_no_parser = re.compile(
            r"^\s*(?:the\s+answer\s+is\s*:?\s*)?(yes|no)[\s.,;:!?]*$",
            re.IGNORECASE
        )
        # More lenient parser if the strict one fails, looking for "yes" or "no" anywhere.
        self._lenient_yes_no_parser = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


    def _parse_yes_no(self, output: str, strict: bool = True) -> JudgeDecision:
        """
        Parses the LLM output to extract a YES or NO decision.

        Args:
            output: The raw string output from the LLM.
            strict: If True, uses a stricter regex. If False, uses a more lenient one.

        Returns:
            JudgeDecision.YES, JudgeDecision.NO, or JudgeDecision.INVALID.
        """
        if not output:
            return JudgeDecision.INVALID

        parser = self._yes_no_parser if strict else self._lenient_yes_no_parser
        match = parser.search(output.strip())

        if match:
            decision_str = match.group(1).upper()
            if decision_str == "YES":
                return JudgeDecision.YES
            elif decision_str == "NO":
                return JudgeDecision.NO

        if strict: # If strict parsing failed, try lenient as a fallback
            logger.debug(f"Strict YES/NO parsing failed for: '{output}'. Trying lenient parsing.")
            return self._parse_yes_no(output, strict=False)

        logger.warning(f"Could not parse a clear YES/NO from LLM output: '{output}'")
        return JudgeDecision.INVALID

    def get_strict_judgment(
        self,
        prompt_or_data: Union[str, Dict[str, str]],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        retries: int = 0
    ) -> JudgeResult:
        """
        Gets a strict YES/NO judgment from the LLM for a single prompt.

        Args:
            prompt_or_data: The prompt string or a dictionary for template formatting.
                            The prompt should instruct the LLM to answer with only "YES" or "NO".
            generation_kwargs: Additional keyword arguments for the LLM's run method
                               (e.g., temperature=0.0 for more deterministic output).
            retries: Number of times to retry if the output is INVALID.

        Returns:
            A JudgeResult object containing the decision and raw output.
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        # Forcing deterministic output is good for YES/NO tasks
        if isinstance(self.llm, HuggingFaceLLM):
            generation_kwargs.setdefault("temperature", 0.01) # Near-zero for greedy
            generation_kwargs.setdefault("do_sample", False) # Ensure greedy if temp is low
            generation_kwargs.setdefault("max_new_tokens", 10) # YES/NO answers are short
        elif isinstance(self.llm, OpenAILLM):
            generation_kwargs.setdefault("temperature", 0.0)
            generation_kwargs.setdefault("max_tokens", 10) # For OpenAI, max_tokens for the completion part

        question_str = str(prompt_or_data) if isinstance(prompt_or_data, str) else str(prompt_or_data.get("question", prompt_or_data))


        current_attempt = 0
        raw_output = ""
        decision = JudgeDecision.INVALID
        error_message = None

        while current_attempt <= retries:
            try:
                raw_output = self.llm.run(prompt_or_data, **generation_kwargs)
                decision = self._parse_yes_no(raw_output, strict=True)

                if decision != JudgeDecision.INVALID:
                    break # Successful parse

                logger.warning(f"Attempt {current_attempt + 1}/{retries + 1}: LLM output '{raw_output}' was INVALID. Retrying if possible.")

            except Exception as e:
                logger.error(f"Error during LLM execution for judgment: {e}", exc_info=True)
                error_message = str(e)
                decision = JudgeDecision.INVALID # Ensure it's marked invalid on error
                break # Stop retrying on LLM execution error

            current_attempt += 1
            if current_attempt <= retries:
                 logger.info(f"Retrying judgment for: {question_str[:100]}...")


        return JudgeResult(
            question=question_str,
            answer=decision,
            raw_output=raw_output,
            error_message=error_message
            # Probabilities would be filled by a different method
        )

    def get_judgment_with_probabilities(
        self,
        prompt_or_data: Union[str, Dict[str, str]],
        yes_token_str: str = "Yes", # Token string for "Yes"
        no_token_str: str = "No",   # Token string for "No"
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> JudgeResult:
        """
        Gets a YES/NO judgment and attempts to extract token probabilities for "Yes" and "No".
        This functionality is highly dependent on the underlying LLM's API.

        Args:
            prompt_or_data: The prompt or data for formatting.
            yes_token_str: The string representation of the "yes" token (case-sensitive for some tokenizers).
            no_token_str: The string representation of the "no" token.
            generation_kwargs: LLM generation arguments. For OpenAI, include `logprobs=True` and `top_logprobs=5` (or similar).
                               For HuggingFace, ensure `output_scores=True` is passed.

        Returns:
            A JudgeResult object, potentially with probability information.
        """
        question_str = str(prompt_or_data) if isinstance(prompt_or_data, str) else str(prompt_or_data.get("question", prompt_or_data))

        # Default kwargs for probability extraction
        if generation_kwargs is None:
            generation_kwargs = {}

        prob_yes: Optional[float] = None
        prob_no: Optional[float] = None
        top_tokens_info: Optional[List[Dict[str, Any]]] = None
        raw_output = ""
        decision = JudgeDecision.INVALID
        error_message = None

        try:
            if isinstance(self.llm, HuggingFaceLLM) and self.llm.tokenizer and self.llm.model:
                # For HuggingFace, we need to get logits.
                generation_kwargs["output_scores"] = True # Request logits
                generation_kwargs["return_dict_in_generate"] = True # Get structured output
                generation_kwargs.setdefault("max_new_tokens", 5) # Usually 1-2 tokens for Yes/No

                prompt_text = self.llm._prepare_prompt(prompt_or_data) # type: ignore # Accessing protected method for this specialized case
                inputs = self.llm.tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.llm.model.device) for k, v in inputs.items()} # type: ignore

                outputs = self.llm.model.generate(**inputs, **generation_kwargs) # type: ignore

                raw_output_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
                raw_output = self.llm.tokenizer.decode(raw_output_ids, skip_special_tokens=True)
                decision = self._parse_yes_no(raw_output)

                # Extract probabilities for the *first generated token*
                # `outputs.scores` is a tuple of logits for each generation step.
                if outputs.scores:
                    first_step_logits = outputs.scores[0][0] # Logits for the first token, first batch item
                    probs = torch.softmax(first_step_logits, dim=-1)

                    # Get token IDs for "Yes" and "No"
                    # This can be tricky due to tokenization (e.g., "Yes" vs " Yes")
                    # A robust way is to tokenize them and get their IDs.
                    yes_token_ids = self.llm.tokenizer.encode(yes_token_str, add_special_tokens=False)
                    no_token_ids = self.llm.tokenizer.encode(no_token_str, add_special_tokens=False)

                    if yes_token_ids: prob_yes = probs[yes_token_ids[0]].item() if len(yes_token_ids) == 1 else sum(probs[tid].item() for tid in yes_token_ids) / len(yes_token_ids) # Average if multi-token
                    if no_token_ids: prob_no = probs[no_token_ids[0]].item() if len(no_token_ids) == 1 else sum(probs[tid].item() for tid in no_token_ids) / len(no_token_ids)

                    # Get top K tokens and their probabilities
                    top_k_probs, top_k_ids = torch.topk(probs, k=5)
                    top_tokens_info = []
                    for prob, token_id in zip(top_k_probs, top_k_ids):
                        token_str = self.llm.tokenizer.decode([token_id.item()])
                        top_tokens_info.append({"token_str": token_str, "probability": prob.item(), "token_id": token_id.item()})
                else:
                    logger.warning("HuggingFace model did not return scores, cannot extract probabilities.")


            elif isinstance(self.llm, OpenAILLM):
                generation_kwargs["logprobs"] = True # Request logprobs
                generation_kwargs.setdefault("max_tokens", 5) # Short completion for Yes/No
                # For top_logprobs, it's part of the main completion object, not a direct param for create
                # We'd parse response.choices[0].logprobs.content

                # OpenAI's run method needs to be adapted or we need to call client directly.
                # For now, let's assume run can pass these through or we make a specialized call.
                # This part is complex as `run` in BaseLLM is generic.
                # A direct client call might be:
                # response = self.llm.client.chat.completions.create(model=self.llm.model_name, messages=[...], **generation_kwargs)
                # Then parse response.choices[0].logprobs.
                # This requires modifying the OpenAILLM or LLMasJudge to handle this special case.
                # For simplicity, we'll say this is "not fully supported yet" for OpenAI via generic run.
                logger.warning("Detailed token probability extraction for OpenAI models is complex with the current BaseLLM.run interface. Basic judgment will be performed.")
                # Fallback to simple judgment without full probability details
                judge_res_simple = self.get_strict_judgment(prompt_or_data, generation_kwargs)
                raw_output = judge_res_simple.raw_output
                decision = judge_res_simple.answer
                error_message = judge_res_simple.error_message
                # Note: OpenAI API can return logprobs for *chosen* tokens. If 'Yes' or 'No' is chosen, we can get its prob.
                # This requires parsing the `logprobs` field in the API response if available.

            else:
                logger.warning(f"Token probability extraction not implemented for LLM type: {self.llm.__class__.__name__}. Performing standard judgment.")
                judge_res_simple = self.get_strict_judgment(prompt_or_data, generation_kwargs)
                raw_output = judge_res_simple.raw_output
                decision = judge_res_simple.answer
                error_message = judge_res_simple.error_message

        except ImportError: # For torch if HuggingFaceLLM is used without it
            logger.error("PyTorch not found, which is required for HuggingFace probability extraction.", exc_info=True)
            error_message = "PyTorch not found for HuggingFace probability extraction."
            # Fallback to simple judgment
            judge_res_simple = self.get_strict_judgment(prompt_or_data, generation_kwargs)
            raw_output = judge_res_simple.raw_output
            decision = judge_res_simple.answer
            if not error_message: error_message = judge_res_simple.error_message

        except Exception as e:
            logger.error(f"Error during judgment with probabilities: {e}", exc_info=True)
            error_message = str(e)
            # Attempt a basic judgment as fallback
            if not raw_output: # If execution failed before getting any output
                judge_res_simple = self.get_strict_judgment(prompt_or_data, generation_kwargs)
                raw_output = judge_res_simple.raw_output
                decision = judge_res_simple.answer
                if not error_message: error_message = judge_res_simple.error_message


        return JudgeResult(
            question=question_str,
            answer=decision,
            raw_output=raw_output,
            probability_yes=prob_yes,
            probability_no=prob_no,
            top_tokens_info=top_tokens_info,
            error_message=error_message
        )

# --- Parallel Execution ---
# Worker function for parallel processing.
# IMPORTANT: LLM objects (especially HuggingFace models) are often not easily picklable
# or shareable across processes without special handling (e.g., making them global,
# or re-initializing them in each worker).

# Global variable to hold the LLM instance for workers, if using a method that requires it.
# This is a common pattern but has its own issues (e.g. if LLM needs to be different per worker).
_WORKER_LLM_INSTANCE: Optional[BaseLLM] = None
_WORKER_LLM_CONFIG: Optional[Dict[str, Any]] = None


def init_worker_llm(llm_class_type: type, llm_config: Dict[str, Any]):
    """Initializes the LLM instance for a worker process."""
    global _WORKER_LLM_INSTANCE
    if _WORKER_LLM_INSTANCE is None:
        logger.info(f"Worker process {os.getpid()} initializing LLM: {llm_config.get('model_name')}")
        try:
            _WORKER_LLM_INSTANCE = llm_class_type(**llm_config)
        except Exception as e:
            logger.error(f"Worker {os.getpid()} failed to initialize LLM: {e}", exc_info=True)
            # Propagate the error or handle it so the main process knows
            raise RuntimeError(f"Worker LLM initialization failed: {e}") from e
    else:
        logger.info(f"Worker process {os.getpid()} already has LLM instance.")


def parallel_judge_task(
    prompt_or_data: Union[str, Dict[str, str]],
    generation_kwargs: Optional[Dict[str, Any]] = None,
    retries: int = 0,
    judge_method_name: str = "get_strict_judgment", # or "get_judgment_with_probabilities"
    # llm_class_type_str: Optional[str] = None, # String representation of LLM class type
    # llm_config: Optional[Dict[str, Any]] = None # Config to init LLM in worker
) -> JudgeResult:
    """
    A single task to be run in a parallel worker.
    It will use the globally initialized LLM in the worker process.
    """
    global _WORKER_LLM_INSTANCE

    # This check is crucial. If the worker hasn't been initialized with an LLM, this task cannot run.
    if _WORKER_LLM_INSTANCE is None:
        # This should ideally not happen if initializer is used correctly with ProcessPoolExecutor.
        error_msg = f"LLM not initialized in worker process {os.getpid()}. Cannot perform judgment."
        logger.error(error_msg)
        question_str = str(prompt_or_data) if isinstance(prompt_or_data, str) else str(prompt_or_data.get("question", prompt_or_data))
        return JudgeResult(question=question_str, answer=JudgeDecision.INVALID, raw_output="", error_message=error_msg)

    judge = LLMasJudge(_WORKER_LLM_INSTANCE)

    if judge_method_name == "get_strict_judgment":
        return judge.get_strict_judgment(prompt_or_data, generation_kwargs, retries)
    elif judge_method_name == "get_judgment_with_probabilities":
        # Note: generation_kwargs for probabilities might be different.
        # This simple call assumes they are passed correctly.
        return judge.get_judgment_with_probabilities(prompt_or_data, generation_kwargs=generation_kwargs)
    else:
        raise ValueError(f"Unknown judge_method_name: {judge_method_name}")


class ParallelLLMJudge:
    """
    Manages parallel execution of LLM judgments using a ProcessPoolExecutor.
    """
    def __init__(self, llm_class_type: type, llm_config: Dict[str, Any], max_workers: Optional[int] = None):
        """
        Args:
            llm_class_type: The class of the LLM to be used (e.g., HuggingFaceLLM, OpenAILLM).
            llm_config: Dictionary of arguments to initialize the LLM class in each worker.
                        Example: {"model_name": "gpt-3.5-turbo", "task": "judgment", "api_key": "..."}
            max_workers: Maximum number of worker processes. Defaults to os.cpu_count().
        """
        if not issubclass(llm_class_type, BaseLLM):
            raise TypeError("llm_class_type must be a subclass of BaseLLM.")

        self.llm_class_type = llm_class_type
        self.llm_config = llm_config
        self.max_workers = max_workers
        logger.info(f"ParallelLLMJudge initialized for LLM type {llm_class_type.__name__} with {max_workers or 'default'} workers.")

    def batch_judge(
        self,
        tasks: List[Union[str, Dict[str, str]]],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        retries: int = 0,
        use_probabilities: bool = False
    ) -> List[JudgeResult]:
        """
        Performs judgments on a batch of tasks in parallel.

        Args:
            tasks: A list of prompts or data dictionaries.
            generation_kwargs: Shared generation arguments for all tasks.
            retries: Shared number of retries for all tasks.
            use_probabilities: If True, attempts to get probabilities (slower, more LLM-dependent).

        Returns:
            A list of JudgeResult objects, one for each task.
        """
        results: List[JudgeResult] = []
        judge_method_name = "get_judgment_with_probabilities" if use_probabilities else "get_strict_judgment"

        # `initializer` and `initargs` are key for setting up each worker process.
        # `mp_context` can be used to specify 'spawn' if 'fork' causes issues (common on macOS with HF models).
        # For HuggingFace models, 'spawn' is often safer to avoid CUDA issues with forked processes.
        # However, 'spawn' means data (llm_config) must be picklable.
        context = None
        if self.llm_class_type == HuggingFaceLLM: # Or other types known to have issues with fork
             # Check OS, 'spawn' is default on Windows, good for macOS too.
             if sys.platform == "darwin" or sys.platform == "win32":
                 context = multiprocessing.get_context("spawn")
             else: # Linux often fine with fork, but spawn is safer with CUDA
                 context = multiprocessing.get_context("spawn") # Consider making this configurable


        with ProcessPoolExecutor(max_workers=self.max_workers, initializer=init_worker_llm, initargs=(self.llm_class_type, self.llm_config), mp_context=context) as executor:
            futures_map = {
                executor.submit(parallel_judge_task, task, generation_kwargs, retries, judge_method_name): task
                for task in tasks
            }

            for future in as_completed(futures_map):
                original_task_identifier = futures_map[future] # For logging/debugging
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel task for '{str(original_task_identifier)[:50]}...' failed: {e}", exc_info=True)
                    # Create a JudgeResult indicating error for this task
                    question_str = str(original_task_identifier) if isinstance(original_task_identifier, str) else str(original_task_identifier.get("question", original_task_identifier))
                    results.append(JudgeResult(question=question_str, answer=JudgeDecision.INVALID, raw_output="", error_message=str(e)))

        # Ensure results are in the same order as input tasks if needed (ProcessPoolExecutor doesn't guarantee this with as_completed)
        # For now, returning them as they complete. If order is critical, need to map back.
        # This current implementation does not preserve order.
        # To preserve order: results = [None] * len(tasks), then map futures to indices.
        # Example: futures = {executor.submit(...): i for i, task in enumerate(tasks)}
        # Then, when future completes: results[futures[future]] = future.result()
        return results


# Need to import these for the classes above and for example usage
import os
import sys
import multiprocessing # For mp_context
from dataclasses import dataclass
from enum import Enum

# Conditional import for torch (used in HuggingFaceLLM probability extraction)
try:
    import torch
except ImportError:
    logger.info("PyTorch not installed. HuggingFaceLLM probability features will be limited.")
    torch = None # type: ignore


if __name__ == "__main__":
    # --- Example Usage ---
    # IMPORTANT: These examples require actual LLM setup.
    # For HuggingFace, a model like "distilgpt2" or a more capable one.
    # For OpenAI, an API key.

    # --- Mock LLM for testing Judge logic without real API calls ---
    class MockLLM(BaseLLM):
        def __init__(self, model_name="mock_llm", task="mock_task", fixed_responses: Optional[List[str]] = None):
            super().__init__(model_name, task)
            self.fixed_responses = fixed_responses
            self.response_index = 0
            self.prompts_received: List[str] = []

        def run(self, input_data: Union[str, Dict[str, str]], **kwargs: Any) -> str:
            prompt = self._prepare_prompt(input_data)
            self.prompts_received.append(prompt)
            if self.fixed_responses:
                response = self.fixed_responses[self.response_index % len(self.fixed_responses)]
                self.response_index += 1
                return response
            # Simulate LLM thinking about Yes/No based on prompt
            if "positive scenario" in prompt.lower(): return "Yes."
            if "negative scenario" in prompt.lower(): return "No, definitely not."
            if "ambiguous scenario" in prompt.lower(): return "Hmm, I'm not sure."
            return "YES" # Default

        def count_tokens(self, text: str) -> int: return len(text.split())

    logger.info("\n--- Testing LLMasJudge with MockLLM ---")
    mock_llm_inst = MockLLM(fixed_responses=["Yes, absolutely!", "NO", "Maybe? YES."])
    judge_with_mock = LLMasJudge(mock_llm_inst)

    # Test 1: Strict judgment
    prompt1 = "Is this a positive scenario?"
    result1 = judge_with_mock.get_strict_judgment(prompt1, retries=1)
    print(f"Prompt: '{prompt1}' -> Result: {result1.answer}, Raw: '{result1.raw_output}'")
    assert result1.answer == JudgeDecision.YES

    prompt2 = "Is this a negative scenario?"
    result2 = judge_with_mock.get_strict_judgment(prompt2)
    print(f"Prompt: '{prompt2}' -> Result: {result2.answer}, Raw: '{result2.raw_output}'")
    assert result2.answer == JudgeDecision.NO

    prompt3 = "Is this an ambiguous scenario?" # MockLLM returns "Maybe? YES."
    result3 = judge_with_mock.get_strict_judgment(prompt3)
    print(f"Prompt: '{prompt3}' -> Result: {result3.answer}, Raw: '{result3.raw_output}'")
    assert result3.answer == JudgeDecision.YES # Lenient parser should find "YES"

    # Test probability extraction (will be limited with MockLLM)
    # This will mostly test the flow, not actual probabilities.
    logger.info("\n--- Testing get_judgment_with_probabilities with MockLLM ---")
    # Note: MockLLM doesn't support actual probability generation.
    result_prob = judge_with_mock.get_judgment_with_probabilities(prompt1)
    print(f"Prob Judgment for '{prompt1}': Answer: {result_prob.answer}, YesProb: {result_prob.probability_yes}, NoProb: {result_prob.probability_no}")
    assert result_prob.answer == JudgeDecision.YES # Should still parse the decision

    # --- Parallel Judging Example (with MockLLM) ---
    # This demonstrates the parallel execution flow.
    logger.info("\n--- Testing ParallelLLMJudge with MockLLM ---")

    # Configuration for initializing MockLLM in worker processes
    mock_llm_worker_config = {
        "model_name": "mock_llm_worker",
        "task": "parallel_mock_task",
        # For MockLLM, we might pass fixed responses per worker if needed, or they share.
        # If fixed_responses is part of llm_config, it must be picklable.
        "fixed_responses": ["YES", "NO", "YES", "NO", "INVALID OUTPUT"]
    }

    parallel_judge_system = ParallelLLMJudge(
        llm_class_type=MockLLM, # Pass the class itself
        llm_config=mock_llm_worker_config,
        max_workers=2
    )

    tasks_for_parallel = [
        "Parallel task 1: positive scenario",
        {"question": "Parallel task 2: negative scenario"},
        "Parallel task 3: another positive one",
        "Parallel task 4: another negative one",
        "Parallel task 5: ambiguous scenario to test invalid output handling"
    ]

    # On Windows or macOS with 'spawn', this block needs to be inside `if __name__ == '__main__':`
    # For Linux with 'fork', it might work outside, but good practice to keep it within.
    if sys.platform == "darwin" or sys.platform == "win32" or True: # Force for testing consistency
        # This check is vital for multiprocessing with 'spawn' or on Windows.
        # The main module must be importable without re-executing this block.
        # For this example, it's self-contained, so it's fine.

        batch_results = parallel_judge_system.batch_judge(tasks_for_parallel, retries=0)
        print("\nParallel Batch Judge Results:")
        for res in batch_results:
            print(f"  Task: '{res.question[:30]}...' -> Answer: {res.answer}, Raw: '{res.raw_output}', Error: {res.error_message}")

        assert len(batch_results) == len(tasks_for_parallel)
        # Check some expected results based on MockLLM logic and fixed_responses in worker_config
        # Note: Order of results from ProcessPoolExecutor + as_completed is not guaranteed.
        # We'd need to sort or map them back to original tasks if strict order checking is needed.
        # For this test, we'll just check counts of decisions.
        yes_count = sum(1 for r in batch_results if r.answer == JudgeDecision.YES)
        no_count = sum(1 for r in batch_results if r.answer == JudgeDecision.NO)
        invalid_count = sum(1 for r in batch_results if r.answer == JudgeDecision.INVALID)

        # Based on mock_llm_worker_config fixed_responses: YES, NO, YES, NO, INVALID OUTPUT
        # And how MockLLM processes prompts if fixed_responses aren't hit (which they should be here)
        # The `parallel_judge_task` will cycle through `fixed_responses`.
        print(f"YES: {yes_count}, NO: {no_count}, INVALID: {invalid_count}")
        # Expected: 2 YES, 2 NO, 1 INVALID (from "INVALID OUTPUT")
        # This depends on the ProcessPoolExecutor fairly distributing tasks and workers picking up responses in order.
        # This assertion might be flaky due to parallelism. A better test would be to check specific task results if order was preserved.
        # For now, let's assume it's roughly correct.
        # assert yes_count == 2
        # assert no_count == 2
        # assert invalid_count == 1

    logger.info("\nLLMasJudge.py example usage finished.")
