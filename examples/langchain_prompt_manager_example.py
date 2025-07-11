# Example script for src/langchain_core/langchain_prompt_manager.py

import sys
import os
import logging

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.langchain_core.langchain_prompt_manager import LangChainPromptManager
except ImportError:
    print("Failed to import LangChainPromptManager. Ensure 'src' is in PYTHONPATH.")
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def run_prompt_manager_example():
    logger.info("--- Starting LangChainPromptManager Example ---")

    prompt_manager = LangChainPromptManager()
    logger.info("LangChainPromptManager initialized.")

    # --- 1. Create a standard PromptTemplate (conceptual) ---
    prompt_name_story = "story_generator"
    story_template_str = "Write a short story about a {adjective} {noun} who discovers a {magical_item}."
    story_input_vars = ["adjective", "noun", "magical_item"]

    logger.info(f"\n1. Creating conceptual 'prompt' template: '{prompt_name_story}'")
    try:
        prompt_manager.create_template(
            name=prompt_name_story,
            template_str=story_template_str,
            input_variables=story_input_vars,
            template_type="prompt",
            partial_variables={"magical_item": "glowing orb"} # Example of partial variable
        )
        logger.info(f"  Template '{prompt_name_story}' conceptually created and stored.")
        retrieved_story_template = prompt_manager.get_template(prompt_name_story)
        logger.info(f"  Retrieved conceptual template: {retrieved_story_template}")
    except Exception as e:
        logger.error(f"  Error creating '{prompt_name_story}': {e}", exc_info=True)

    # --- 2. Create a ChatPromptTemplate (conceptual) ---
    # In a real ChatPromptTemplate, template_str might be a list of message tuples or objects
    prompt_name_chat = "simple_chat_greeting"
    chat_template_str = "Hello, {user_name}! You are chatting with {ai_name}." # Simplified for conceptual
    # For a real ChatPromptTemplate, this might be:
    # [("system", "You are {ai_name}."), ("human", "Hello, I am {user_name}.")]
    chat_input_vars = ["user_name", "ai_name"]

    logger.info(f"\n2. Creating conceptual 'chat' template: '{prompt_name_chat}'")
    try:
        prompt_manager.create_template(
            name=prompt_name_chat,
            template_str=chat_template_str, # This would be handled differently by actual ChatPromptTemplate
            input_variables=chat_input_vars,
            template_type="chat"
        )
        logger.info(f"  Template '{prompt_name_chat}' conceptually created and stored.")
    except Exception as e:
        logger.error(f"  Error creating '{prompt_name_chat}': {e}", exc_info=True)

    # --- 3. List all templates ---
    logger.info(f"\n3. Listing all templates in manager:")
    all_templates = prompt_manager.list_templates()
    logger.info(f"  Managed templates: {all_templates}")

    # --- 4. Format the 'story_generator' prompt (conceptual) ---
    logger.info(f"\n4. Formatting '{prompt_name_story}' (conceptual):")
    story_format_values = {"adjective": "curious", "noun": "cat"}
    # 'magical_item' is partially filled
    try:
        formatted_story_prompt = prompt_manager.format_prompt(
            name=prompt_name_story,
            **story_format_values
        )
        logger.info(f"  Formatted story prompt (conceptual):\n    '{formatted_story_prompt}'")
    except Exception as e:
        logger.error(f"  Error formatting '{prompt_name_story}': {e}", exc_info=True)

    # --- 5. Format the 'simple_chat_greeting' prompt (conceptual) ---
    logger.info(f"\n5. Formatting '{prompt_name_chat}' (conceptual):")
    chat_format_values = {"user_name": "Bob", "ai_name": "AssistantGPT"}
    try:
        # Note: Real ChatPromptTemplate.format_prompt returns a ChatPromptValue, not a string.
        # Our conceptual version returns a string for simplicity here.
        formatted_chat_prompt = prompt_manager.format_prompt(
            name=prompt_name_chat,
            **chat_format_values
        )
        logger.info(f"  Formatted chat prompt (conceptual):\n    '{formatted_chat_prompt}'")
    except Exception as e:
        logger.error(f"  Error formatting '{prompt_name_chat}': {e}", exc_info=True)

    # --- 6. Save and Load a template (conceptual) ---
    conceptual_filepath = "conceptual_story_template.json" # Not actually written to disk
    logger.info(f"\n6. Saving '{prompt_name_story}' to '{conceptual_filepath}' (conceptual):")
    try:
        prompt_manager.save_template(name=prompt_name_story, filepath=conceptual_filepath)
        logger.info(f"  Conceptual save complete.")
    except Exception as e:
        logger.error(f"  Error saving '{prompt_name_story}': {e}", exc_info=True)

    logger.info(f"\n7. Loading template from '{conceptual_filepath}' (conceptual):")
    loaded_template_name = "loaded_story_prompt"
    try:
        loaded_template = prompt_manager.load_template(
            filepath=conceptual_filepath,
            name_for_manager=loaded_template_name
        )
        logger.info(f"  Template conceptually loaded as '{loaded_template_name}'.")
        logger.info(f"  Content of loaded (conceptual): {loaded_template}")

        # Try formatting the loaded template
        formatted_loaded_prompt = prompt_manager.format_prompt(
            name=loaded_template_name,
            adjective="sleepy",
            noun="dragon"
            # 'magical_item' would use its partial value from original creation if preserved
        )
        logger.info(f"  Formatted loaded prompt (conceptual):\n    '{formatted_loaded_prompt}'")

    except Exception as e:
        logger.error(f"  Error during conceptual load/format of loaded template: {e}", exc_info=True)

    logger.info(f"\n8. Templates in manager after load: {prompt_manager.list_templates()}")

    # --- Error case: Template not found ---
    logger.info(f"\n9. Attempting to format a non-existent template:")
    try:
        prompt_manager.format_prompt(name="non_existent_template", text="test")
    except KeyError as e:
        logger.warning(f"  Caught expected error for non-existent template: {e}")
    except Exception as e:
        logger.error(f"  Unexpected error for non-existent template: {e}", exc_info=True)

    logger.info("\n--- LangChainPromptManager Example Finished ---")

if __name__ == "__main__":
    run_prompt_manager_example()
