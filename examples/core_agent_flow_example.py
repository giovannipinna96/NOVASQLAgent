# Example script for src/langchain_core/langgraph_flow.py (Optional Advanced)

import sys
import os
import logging
import json # For pretty printing dicts

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.langchain_core.langchain_llm import LangChainLLM
    from src.langchain_core.langchain_prompt_manager import LangChainPromptManager
    from src.langchain_core.langchain_memory import LangChainMemory
    from src.langchain_core.langgraph_flow import LangGraphFlow
except ImportError:
    print("Failed to import core modules. Ensure 'src' is in PYTHONPATH.")
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper to pretty print state ---
def pprint_state(state_dict, title="State"):
    """Pretty prints a dictionary, typically the agent state."""
    logger.info(f"--- {title} ---")
    try:
        # Convert any non-serializable items to string for robust printing
        # For this conceptual example, most things are strings already.
        serializable_state = {k: str(v) for k, v in state_dict.items()}
        logger.info(json.dumps(serializable_state, indent=2))
    except Exception as e:
        logger.error(f"Could not pretty print state due to: {e}")
        logger.info(str(state_dict)) # Fallback to simple string print
    logger.info("--- End State ---")


def run_langgraph_flow_example():
    logger.info("--- Starting LangGraphFlow Conceptual Example ---")

    # 1. Initialize Conceptual Managers
    # These use the conceptual implementations from src.langchain_core
    logger.info("\n1. Initializing Conceptual Managers...")

    llm_manager = LangChainLLM(
        model_name_or_path="conceptual-chat-model",
        model_type="transformers" # Keep it simple for conceptual flow
    )

    prompt_manager = LangChainPromptManager()
    # Create a default prompt template that the LangGraphFlow will use
    # This template expects 'input' (current user query) and 'chat_history_strings'
    agent_prompt_name = "conversational_agent_prompt"
    prompt_manager.create_template(
        name=agent_prompt_name,
        template_str=(
            "You are a helpful AI assistant.\n"
            "Current conversation:\n{chat_history_strings}\n\n"
            "User: {input}\n"
            "AI:"
        ),
        input_variables=["input", "chat_history_strings"],
        template_type="prompt" # Using a simple prompt template for this conceptual example
    )

    memory_manager = LangChainMemory(
        memory_type="buffer", # Use buffer memory for conversation history
        memory_key="chat_history" # This key is used internally by LangChainMemory
    )
    logger.info("Conceptual LLM, Prompt, and Memory Managers initialized.")

    # 2. Initialize LangGraphFlow
    logger.info("\n2. Initializing LangGraphFlow...")
    # The LangGraphFlow's default_prompt_name should match the one created above
    agent_flow = LangGraphFlow(
        llm_manager=llm_manager,
        prompt_manager=prompt_manager,
        memory_manager=memory_manager,
        default_prompt_name=agent_prompt_name
    )
    logger.info("LangGraphFlow conceptually initialized and graph built.")

    # 3. Execute the Flow - First Interaction
    logger.info("\n--- First Interaction ---")
    user_input_1 = "Hello! What is the capital of France?"
    logger.info(f"User Input: '{user_input_1}'")

    final_state_1 = agent_flow.execute_flow(initial_user_input=user_input_1)

    logger.info(f"\nFinal Response to User (1): {final_state_1.get('final_response')}")
    # pprint_state(final_state_1, title="Final State After First Interaction")

    # Log current memory after first interaction (as managed by LangGraphFlow's memory_manager)
    logger.info(f"Memory after 1st interaction:\n{memory_manager.get_history(format_type='string')}")


    # 4. Execute the Flow - Second Interaction (to see memory in action)
    logger.info("\n--- Second Interaction ---")
    user_input_2 = "Thanks! And what is its population?"
    logger.info(f"User Input: '{user_input_2}'")

    final_state_2 = agent_flow.execute_flow(initial_user_input=user_input_2)

    logger.info(f"\nFinal Response to User (2): {final_state_2.get('final_response')}")
    # pprint_state(final_state_2, title="Final State After Second Interaction")

    logger.info(f"Memory after 2nd interaction:\n{memory_manager.get_history(format_type='string')}")

    # 5. Execute the Flow - Third Interaction
    logger.info("\n--- Third Interaction ---")
    user_input_3 = "Tell me a joke."
    logger.info(f"User Input: '{user_input_3}'")

    final_state_3 = agent_flow.execute_flow(initial_user_input=user_input_3)
    logger.info(f"\nFinal Response to User (3): {final_state_3.get('final_response')}")
    logger.info(f"Memory after 3rd interaction:\n{memory_manager.get_history(format_type='string')}")


    logger.info("\n--- LangGraphFlow Conceptual Example Finished ---")

if __name__ == "__main__":
    run_langgraph_flow_example()
