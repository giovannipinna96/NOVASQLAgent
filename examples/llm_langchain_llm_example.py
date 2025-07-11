# Example script for src/langchain_core/langchain_llm.py

import sys
import os
import logging

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.langchain_core.langchain_llm import LangChainLLM
except ImportError:
    print("Failed to import LangChainLLM. Ensure 'src' is in PYTHONPATH.")
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    sys.exit(1)

# Configure logging to see output from the LangChainLLM class
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def run_llm_example():
    logger.info("--- Starting LangChainLLM Example ---")

    # --- Configuration for Transformers-based model (conceptual) ---
    model_id_transformers = "gpt2" # A common small model
    logger.info(f"1. Initializing LangChainLLM with Transformers (conceptual): model='{model_id_transformers}'")

    try:
        llm_transformers = LangChainLLM(
            model_name_or_path=model_id_transformers,
            model_type="transformers", # Explicitly use transformers
            device="cpu", # Conceptual device
            max_new_tokens=50,
            temperature=0.2,
            model_kwargs={"trust_remote_code": True}, # Example model kwarg
            tokenizer_kwargs={}
        )
        logger.info(f"LangChainLLM (Transformers) conceptually initialized for '{model_id_transformers}'.")
        logger.info(f"  Conceptual Model: {llm_transformers.model}")
        logger.info(f"  Conceptual Tokenizer: {llm_transformers.tokenizer}")

        prompt1 = "Explain the concept of a Large Language Model in one sentence."
        logger.info(f"\n  Generating response for prompt: '{prompt1}'")
        response1 = llm_transformers.generate_response(prompt1, stop=["."])
        logger.info(f"  Conceptual Response (Transformers): {response1}")

        prompt2 = "What is the capital of France?"
        logger.info(f"\n  Generating response for prompt: '{prompt2}' with different runtime params")
        response2 = llm_transformers.generate_response(prompt2, max_new_tokens=10, temperature=0.01)
        logger.info(f"  Conceptual Response (Transformers): {response2}")

    except NotImplementedError as e:
        logger.error(f"  NotImplementedError for Transformers: {e}")
        logger.error("  This is expected if actual library calls are hit without libraries installed.")
    except Exception as e:
        logger.error(f"  An unexpected error occurred with Transformers LLM: {e}", exc_info=True)

    # --- Configuration for vLLM-based model (conceptual) ---
    # Note: vLLM typically requires more substantial models and GPU resources.
    model_id_vllm = "NousResearch/Llama-2-7b-hf" # Example, would require vLLM setup
    logger.info(f"\n2. Initializing LangChainLLM with vLLM (conceptual): model='{model_id_vllm}'")

    try:
        llm_vllm = LangChainLLM(
            model_name_or_path=model_id_vllm,
            model_type="vllm", # Explicitly use vLLM
            max_new_tokens=60,
            temperature=0.3,
            model_kwargs={'tensor_parallel_size': 1, 'trust_remote_code': True} # Example vLLM kwargs
        )
        logger.info(f"LangChainLLM (vLLM) conceptually initialized for '{model_id_vllm}'.")
        logger.info(f"  Conceptual Model (vLLM Engine): {llm_vllm.model}")

        prompt3 = "Write a short poem about a robot."
        logger.info(f"\n  Generating response for prompt (vLLM): '{prompt3}'")
        response3 = llm_vllm.generate_response(prompt3, stop=["\n\n"])
        logger.info(f"  Conceptual Response (vLLM): {response3}")

    except NotImplementedError as e:
        logger.error(f"  NotImplementedError for vLLM: {e}")
        logger.error("  This is expected as vLLM loading is conceptual and may raise this error directly.")
    except ValueError as e: # Catch model_type validation errors etc.
        logger.error(f"  ValueError for vLLM setup: {e}")
    except Exception as e:
        logger.error(f"  An unexpected error occurred with vLLM LLM: {e}", exc_info=True)

    # --- Test PEFT/LoRA loading (conceptual) ---
    logger.info(f"\n3. Initializing LangChainLLM with Transformers and PEFT adapter (conceptual)")
    try:
        llm_peft = LangChainLLM(
            model_name_or_path="gpt2", # Base model
            model_type="transformers",
            use_peft=True,
            peft_model_path="path/to/conceptual/lora_adapter", # Conceptual path
            device="cpu"
        )
        logger.info(f"LangChainLLM (PEFT) conceptually initialized.")
        logger.info(f"  Conceptual Model with PEFT: {llm_peft.model}")

        prompt4 = "What does this LoRA model specialize in?"
        response4 = llm_peft.generate_response(prompt4)
        logger.info(f"  Conceptual Response (PEFT): {response4}")

    except NotImplementedError as e:
        logger.error(f"  NotImplementedError for PEFT: {e}")
    except Exception as e:
        logger.error(f"  An unexpected error occurred with PEFT LLM: {e}", exc_info=True)


    logger.info("\n--- LangChainLLM Example Finished ---")

if __name__ == "__main__":
    # This script is designed to be run to see the conceptual output.
    # It will not perform real LLM operations unless all libraries are installed
    # and the conceptual blocks in langchain_llm.py are filled in.
    run_llm_example()
