"""
This file is responsible for managing prompt template files in the most efficient and structured way possible.
It defines either a dedicated class or a well-organized set of functions to load, parse, format, and display
prompt templates that are stored as plain `.txt` files. The functionality defined here must integrate seamlessly
with the codebase responsible for handling LLMs (e.g., the `LLMmodel.py` module).

OBJECTIVES AND FUNCTIONAL REQUIREMENTS:

1. PROMPT TEMPLATE LOADING:
    - The system must accept a path to a `.txt` file containing a prompt template.
    - The prompt should be read as a plain text string from the file.
    - Templates will contain placeholders in the form of `{key}` for variable insertion.

2. PROMPT FORMATTING:
    - A method or function must be implemented that takes a dictionary as input.
    - This dictionary should be used to dynamically fill in the placeholders inside the prompt template.
    - The formatting system must raise meaningful exceptions if any required keys are missing or if formatting fails.

3. VISUALIZATION & DEBUGGING UTILITIES:
    - Include a method to print a "fancy" overview of:
        • The raw template structure (before formatting).
        • The expected keys to be filled.
    - Provide a function or method that prints the fully rendered prompt after formatting with the input dictionary.

4. INTEGRATION & USABILITY:
    - The implementation should be modular and decoupled enough to integrate directly with LLM-handling modules.
    - Ensure reusability so that prompt templates can be managed and reused across different models and tasks.

5. CODING STYLE AND BEST PRACTICES:
    - All code must follow Pythonic principles and best practices:
        • Full type annotations for all function arguments and return types.
        • Adherence to PEP8 for readability and maintainability.
        • Proper use of exceptions and logging where appropriate.
        • Docstrings and comments should be added for all public methods.
        • Ensure extensibility for future features such as loading from YAML, JSON, or remote sources.

This file forms a critical utility layer for safely and dynamically preparing prompts that will be passed to
Large Language Models. The design should favor clarity, robustness, and ease of use in complex software systems.
"""
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptTemplateManager:
    """
    Manages loading, formatting, and displaying prompt templates from .txt files.
    """

    def __init__(self, template_path: Union[str, Path]):
        """
        Initializes the PromptTemplateManager with a path to a template file.

        Args:
            template_path: Path to the .txt prompt template file.

        Raises:
            FileNotFoundError: If the template file does not exist.
            ValueError: If the template_path is not a .txt file.
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            logger.error(f"Template file not found: {self.template_path}")
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        if not self.template_path.suffix == '.txt':
            logger.error(f"Invalid template file extension: {self.template_path.suffix}. Expected .txt")
            raise ValueError("Template file must be a .txt file.")

        self._raw_template: str = self._load_template()
        self._expected_keys: List[str] = self._extract_keys()

    def _load_template(self) -> str:
        """Loads the prompt template from the file."""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading template file {self.template_path}: {e}")
            raise

    def _extract_keys(self) -> List[str]:
        """Extracts placeholder keys (e.g., {key}) from the raw template."""
        # Using a regex to find all occurrences of {key}
        keys = re.findall(r'\{([^{}]+)\}', self._raw_template)
        # Remove duplicates while preserving order
        unique_keys = []
        for key in keys:
            if key not in unique_keys:
                unique_keys.append(key)
        return unique_keys

    def format_prompt(self, input_dict: Dict[str, str]) -> str:
        """
        Formats the prompt template with the provided dictionary of key-value pairs.

        Args:
            input_dict: A dictionary where keys match the placeholders in the template.

        Returns:
            The formatted prompt string.

        Raises:
            KeyError: If a key expected by the template is missing in input_dict.
            ValueError: If there's an issue during formatting (e.g., unexpected placeholder format).
        """
        missing_keys = [key for key in self._expected_keys if key not in input_dict]
        if missing_keys:
            logger.error(f"Missing keys for formatting: {missing_keys}")
            raise KeyError(f"Missing keys for formatting: {', '.join(missing_keys)}")

        try:
            formatted_prompt = self._raw_template.format_map(input_dict)
            return formatted_prompt
        except ValueError as e: # Handles issues like malformed placeholders if not caught by regex
            logger.error(f"Error formatting prompt: {e}. Check placeholder syntax.")
            raise ValueError(f"Error formatting prompt: {e}. Ensure placeholders are correctly formatted like {{key}}.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during prompt formatting: {e}")
            raise

    def display_template_overview(self) -> None:
        """
        Prints a "fancy" overview of the raw template structure and expected keys.
        """
        print("\n--- Prompt Template Overview ---")
        print(f"Template File: {self.template_path.name}")
        print("\nRaw Template Structure:")
        print("-" * 30)
        print(self._raw_template)
        print("-" * 30)
        print("\nExpected Keys for Formatting:")
        if self._expected_keys:
            for key in self._expected_keys:
                print(f"  - {{{key}}}")
        else:
            print("  (No placeholders found in this template)")
        print("--- End of Overview ---\n")

    def display_rendered_prompt(self, input_dict: Dict[str, str]) -> None:
        """
        Formats the prompt with input_dict and prints the fully rendered prompt.

        Args:
            input_dict: A dictionary for formatting the prompt.
        """
        try:
            rendered_prompt = self.format_prompt(input_dict)
            print("\n--- Rendered Prompt ---")
            print(rendered_prompt)
            print("--- End of Rendered Prompt ---\n")
        except (KeyError, ValueError) as e:
            print(f"Could not render prompt: {e}")

    @property
    def raw_template(self) -> str:
        """Returns the raw template string."""
        return self._raw_template

    @property
    def expected_keys(self) -> List[str]:
        """Returns a list of expected keys for formatting."""
        return self._expected_keys

if __name__ == '__main__':
    # Example Usage (assuming a dummy template file exists)
    # Create a dummy template file for testing
    dummy_template_content = """\
Hello, {name}!
Today is {day}.
How are you feeling, {name}?
This is a test for {test_name}.
"""
    dummy_template_path = Path("dummy_template.txt")
    with open(dummy_template_path, "w", encoding="utf-8") as f:
        f.write(dummy_template_content)

    try:
        manager = PromptTemplateManager(dummy_template_path)

        # Display overview
        manager.display_template_overview()

        # Format and display rendered prompt
        valid_inputs = {"name": "Alice", "day": "Monday", "test_name": "PromptManager"}
        manager.display_rendered_prompt(valid_inputs)

        # Test missing key
        print("Testing with missing key 'day':")
        invalid_inputs = {"name": "Bob", "test_name": "ErrorHandling"}
        try:
            manager.format_prompt(invalid_inputs)
        except KeyError as e:
            print(f"Caught expected error: {e}")

        # Test formatting with all keys
        formatted = manager.format_prompt(valid_inputs)
        assert "Alice" in formatted
        assert "Monday" in formatted
        assert "PromptManager" in formatted
        logger.info("PromptTemplateManager example usage successful.")

    except Exception as e:
        logger.error(f"Error in example usage: {e}")
    finally:
        # Clean up dummy file
        if dummy_template_path.exists():
            dummy_template_path.unlink()
            logger.info(f"Cleaned up {dummy_template_path}")
