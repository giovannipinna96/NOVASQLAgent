"""
This module implements the core logic for building an "LLM-as-a-Judge" system, where a language model (LLM) is used
to evaluate input scenarios or claims and produce a strict binary verdict: YES or NO. This approach is commonly
used in benchmarking, comparative model evaluation, factual consistency testing, or safety-critical decision pipelines.

MODULE GOALS AND SPECIFICATIONS:

1. SYSTEM INTEGRATION:
    - This file must rely on and integrate with existing components from the project:
        • LLM management (see `LLMmodel.py`)
        • Prompt template formatting (see `prompt_template_manager.py`)
        • Memory/conversation history (see `conversation_memory.py`)
    - The system should assume an LLM class that supports `.run()` or similar execution logic, and that can be
      initialized with named models, task, prompt path, and memory options.

2. STRICT YES/NO RESPONSE FUNCTION:
    - Implement a function that takes an LLM instance and a prompt as input, executes the prompt, and returns strictly
      "YES" or "NO" as a response.
    - The system must enforce binary answers. If the output is not exactly "yes" or "no" (case-insensitive),
      it must:
        • Immediately halt execution of that instance
        • Raise a runtime alert or warning
        • Optionally retry or escalate
    - This helps ensure the model's judgment is reliable and aligned to expected structure for downstream pipelines.

3. TOKEN PROBABILITY ANALYSIS:
    - Implement a function to query the LLM in a way that:
        • Extracts the probability distribution over the output tokens
        • Identifies the probability assigned specifically to "yes" and "no" tokens
    - Return a dictionary or data structure containing:
        {
            "yes_token": probability_yes,
            "no_token": probability_no,
            "top_tokens": [...],  # Optional: additional insights
        }
    - This allows interpretation of uncertainty and model confidence in its judgment.

4. PARALLEL EXECUTION PIPELINE:
    - Design a multiprocessing or multithreaded framework to perform judgment evaluation in parallel.
    - The system must:
        • Load the LLM across multiple GPU memory pools or instances (multi-GPU or model sharding compatible)
        • Use a queue/pool of processes to dispatch tasks concurrently
        • Collect all results efficiently

    - Each task should return a structured result such as:
        {
            "question": "...",
            "answer": "YES" or "NO",
            "probability_yes": float,
            "probability_no": float
        }

    - This system must be scalable, fault-tolerant, and configurable for batch evaluation.

5. ENGINEERING AND BEST PRACTICES:
    - Ensure the code follows modern Python best practices:
        • PEP8 style guide compliance
        • Robust type annotations
        • Logging instead of print statements
        • Exception handling for model output, token sampling, and multiprocessing errors
        • Modularity to allow reuse and integration with evaluation or UI layers

    - Consider including the ability to set a temperature, top-k/top-p sampling strategy if required, and deterministic decoding options for strict binary evaluations.

    - All major functions and classes must be documented with clear docstrings explaining inputs, outputs, and side effects.

This file forms the core of a decision-based LLM pipeline and must guarantee precision, consistency, and interpretability
in binary classification decisions. Ideal for evaluation frameworks, fact verification systems, and aligned AI agents.
"""
