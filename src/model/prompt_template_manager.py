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
