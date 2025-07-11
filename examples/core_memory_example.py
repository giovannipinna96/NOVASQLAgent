# Example script for src/langchain_core/langchain_memory.py

import sys
import os
import logging

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.langchain_core.langchain_memory import LangChainMemory
    # For conceptual LLM in summary memory example:
    from src.langchain_core.langchain_llm import LangChainLLM
except ImportError:
    print("Failed to import LangChainMemory or LangChainLLM. Ensure 'src' is in PYTHONPATH.")
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Conceptual LLM for Summary Memory ---
# In a real scenario, this would be a proper LLM instance.
# For this example, we use our conceptual LangChainLLM.
conceptual_llm_for_summary = LangChainLLM(model_name_or_path="conceptual-summary-model", model_type="transformers")

def run_memory_example():
    logger.info("--- Starting LangChainMemory Example ---")

    # --- 1. ConversationBufferMemory (Conceptual) ---
    logger.info("\n1. Testing Conceptual ConversationBufferMemory")
    buffer_memory = LangChainMemory(
        memory_type="buffer",
        memory_key="chat_history", # Standard key
        input_key="user_query",   # Example input key
        human_prefix="User",
        ai_prefix="Bot"
    )

    buffer_memory.add_turn(user_input="Hi Bot, how are you?", ai_response="I'm doing well, User! Thanks for asking.")
    buffer_memory.add_message("This is a system broadcast.", role="system") # Using add_message
    buffer_memory.add_turn(user_input="What can you do?", ai_response="I can remember our conversation.")

    logger.info(f"  Buffer Memory History (string):\n{buffer_memory.get_history(format_type='string')}")
    logger.info(f"  Buffer Memory History (messages list conceptual):\n{buffer_memory.get_history(format_type='messages')}")
    logger.info(f"  Buffer Memory Variables (conceptual):\n{buffer_memory.get_memory_variables({'user_query': 'test'})}")

    buffer_memory.clear_memory()
    logger.info(f"  Buffer Memory History after clear: '{buffer_memory.get_history(format_type='string')}'")

    # --- 2. ConversationBufferWindowMemory (Conceptual) ---
    logger.info("\n2. Testing Conceptual ConversationBufferWindowMemory (k=2)")
    window_memory = LangChainMemory(
        memory_type="buffer_window",
        k=2,
        memory_key="recent_chats",
        return_messages=True # This is often default or implied for message list output
    )
    window_memory.add_turn("User1: First message", "Bot1: First response")
    window_memory.add_turn("User2: Second message", "Bot2: Second response")
    logger.info(f"  Window Memory (k=2) after 2 turns:\n{window_memory.get_history(format_type='string')}")

    window_memory.add_turn("User3: Third message, this should push out the first turn.", "Bot3: Third response")
    logger.info(f"  Window Memory (k=2) after 3 turns (should only contain last 2):\n{window_memory.get_history(format_type='string')}")

    # --- 3. ConversationSummaryMemory (Conceptual) ---
    logger.info("\n3. Testing Conceptual ConversationSummaryMemory")
    try:
        summary_memory = LangChainMemory(
            memory_type="summary",
            llm_instance=conceptual_llm_for_summary, # type: ignore
            memory_key="conversation_summary",
            max_token_limit=50 # Conceptual limit
        )
        summary_memory.add_turn("Human: My favorite color is blue.", "AI: Got it, your favorite color is blue.")
        summary_memory.add_turn("Human: I also like dogs.", "AI: Noted, you like dogs.")
        summary_memory.add_turn("Human: What are my preferences?", "AI: Your favorite color is blue and you like dogs.") # LLM would generate summary

        # The actual history will be a summary string, conceptually
        logger.info(f"  Summary Memory History (string - conceptual summary):\n{summary_memory.get_history(format_type='string')}")
        # In a real summary memory, get_history(format_type="messages") might return a SystemMessage with the summary.
        logger.info(f"  Summary Memory History (messages - conceptual summary as message):\n{summary_memory.get_history(format_type='messages')}")

    except ValueError as ve: # Catch if LLM instance is missing (though we provided conceptual one)
        logger.error(f"  Error initializing summary memory: {ve}")
    except NotImplementedError as nie: # If summary memory itself is not fully fleshed out
        logger.error(f"  NotImplementedError for summary memory: {nie}")


    # --- 4. Conceptual Serialization/Deserialization ---
    logger.info("\n4. Testing Conceptual Save/Load for Buffer Memory")
    buffer_memory_for_save = LangChainMemory(memory_type="buffer", memory_key="saved_history")
    buffer_memory_for_save.add_turn("SaveUser: This will be saved.", "SaveAI: Indeed it will.")
    buffer_memory_for_save.add_turn("SaveUser: And this too.", "SaveAI: Correct.")

    conceptual_save_path = "conceptual_memory.json" # Not actually written to disk

    logger.info(f"  Content before conceptual save:\n{buffer_memory_for_save.get_history(format_type='string')}")
    buffer_memory_for_save.save_memory_to_json(conceptual_save_path)
    logger.info(f"  Memory conceptually saved to {conceptual_save_path}")

    buffer_memory_for_load = LangChainMemory(memory_type="buffer", memory_key="loaded_history")
    logger.info(f"  Memory before conceptual load:\n{buffer_memory_for_load.get_history(format_type='string')}")
    buffer_memory_for_load.load_memory_from_json(conceptual_save_path)
    logger.info(f"  Memory conceptually loaded from {conceptual_save_path}")
    logger.info(f"  Content after conceptual load:\n{buffer_memory_for_load.get_history(format_type='string')}")


    logger.info("\n--- LangChainMemory Example Finished ---")

if __name__ == "__main__":
    run_memory_example()
