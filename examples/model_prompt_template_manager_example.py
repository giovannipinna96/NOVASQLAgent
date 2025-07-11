# examples/model_prompt_template_manager_example.py

import sys
import os
import json # For potentially loading templates from JSON string/file

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.model.prompt_template_manager import PromptTemplateManager
    # from src.model.prompt_template_manager import TemplateError # Potential custom exception
except ImportError:
    print("Warning: Could not import PromptTemplateManager from src.model.prompt_template_manager.")
    print("Using a dummy PromptTemplateManager class for demonstration purposes.")

    class PromptTemplateManager:
        """
        Dummy PromptTemplateManager class for demonstration.
        """
        def __init__(self, template_directory=None, default_templates=None):
            self.templates = {}
            self.template_directory = template_directory
            print(f"Dummy PromptTemplateManager initialized. Template dir: {template_directory}")

            if default_templates:
                self.load_templates(default_templates)
            elif template_directory:
                # In a real scenario, this might scan the directory
                print(f"Would scan '{template_directory}' for templates if this were real.")
                # Let's add a dummy template if a directory is mentioned
                self.templates["dummy_template_from_dir"] = "Hello, {{name}} from directory!"


        def load_templates(self, templates_data, source_name="default"):
            """
            Dummy method to load templates from a dictionary or a file path (simulated).
            templates_data can be a dict or a string path to a JSON/YAML file.
            """
            if isinstance(templates_data, str): # Simulate loading from file
                print(f"\nSimulating loading templates from file: {templates_data}")
                # Dummy content if it were a file
                if templates_data.endswith(".json"):
                    try:
                        # Simulate reading content that would be in the file
                        # For this dummy, let's assume a fixed structure
                        file_content_sim = {
                            "greeting_template": "Hello, {{name}}! Welcome to {{place}}.",
                            "sql_generation_template": "Translate the following natural language query to SQL, considering the schema {{db_schema}}: {{nl_query}}"
                        }
                        loaded_tpls = file_content_sim
                        print(f"Successfully simulated loading from JSON file '{templates_data}'.")
                    except Exception as e:
                        print(f"Error simulating JSON load from '{templates_data}': {e}")
                        return False
                else:
                    print(f"Unsupported dummy file type for source: {templates_data}")
                    return False
            elif isinstance(templates_data, dict):
                loaded_tpls = templates_data
                print(f"\nLoading templates from dictionary source: '{source_name}'")
            else:
                print("\nInvalid template data format. Must be dict or file path string.")
                return False

            for name, template_string in loaded_tpls.items():
                self.templates[name] = template_string
                print(f"Template '{name}' loaded/updated.")
            return True

        def get_template(self, template_name):
            """
            Dummy method to retrieve a raw template string.
            """
            template = self.templates.get(template_name)
            if template:
                print(f"\nRetrieved template '{template_name}': '{template}'")
            else:
                print(f"\nTemplate '{template_name}' not found.")
            return template

        def format_prompt(self, template_name, **kwargs):
            """
            Dummy method to format a prompt using a stored template and provided variables.
            Uses simple string replacement for {{variable_name}}.
            """
            template_string = self.templates.get(template_name)
            if not template_string:
                print(f"\nError: Template '{template_name}' not found for formatting.")
                # raise ValueError(f"Template '{template_name}' not found.") # Or return error string
                return f"Error: Template '{template_name}' not found."

            print(f"\nFormatting prompt for template '{template_name}' with variables: {kwargs}")
            formatted_prompt = template_string
            for key, value in kwargs.items():
                placeholder = "{{" + key + "}}" # or f"{{{key}}}"
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

            # Check for unreplaced placeholders (basic check)
            if "{{" in formatted_prompt and "}}" in formatted_prompt:
                print(f"Warning: Potential unreplaced placeholders in formatted prompt: {formatted_prompt}")

            print(f"Formatted Prompt: '{formatted_prompt}'")
            return formatted_prompt

        def list_available_templates(self):
            """
            Dummy method to list all available template names.
            """
            print("\nAvailable templates:")
            if not self.templates:
                print("No templates loaded.")
                return []
            for name in self.templates.keys():
                print(f"- {name}")
            return list(self.templates.keys())

def main():
    print("--- PromptTemplateManager Module Example ---")

    # Define some initial templates as a dictionary
    initial_templates = {
        "user_query_enhancement": "Original query: {{user_query}}\nEnhanced query based on history ({{history_summary}}):",
        "text_to_sql_main": "### Database Schema:\n{{db_schema}}\n\n### Question:\n{{natural_language_question}}\n\n### SQL Query:\n",
        "error_correction_prompt": "The previous SQL query '{{previous_sql}}' resulted in the error '{{error_message}}'. Please provide a corrected SQL query. Schema: {{db_schema}}"
    }

    # Instantiate PromptTemplateManager
    try:
        # Can be initialized with a directory path (for auto-loading) or with initial templates
        # prompt_manager = PromptTemplateManager(template_directory="./prompt_templates")
        prompt_manager = PromptTemplateManager(default_templates=initial_templates)
    except NameError: # Fallback for dummy
        prompt_manager = PromptTemplateManager(default_templates=initial_templates)


    # Example 1: List available templates
    print("\n[Example 1: List Initial Templates]")
    prompt_manager.list_available_templates()

    # Example 2: Get a raw template
    print("\n[Example 2: Get a Raw Template]")
    raw_template = prompt_manager.get_template("text_to_sql_main")
    # if raw_template:
    #     print(f"Raw 'text_to_sql_main' template: {raw_template}")

    # Example 3: Format a prompt using a template
    print("\n[Example 3: Format 'text_to_sql_main' Template]")
    db_schema_example = "Table users: id, name, email\nTable orders: order_id, user_id, product, amount"
    nl_question_example = "Show me all orders for user with email 'test@example.com'"

    formatted_sql_prompt = prompt_manager.format_prompt(
        template_name="text_to_sql_main",
        db_schema=db_schema_example,
        natural_language_question=nl_question_example
    )
    # print(f"Formatted SQL Generation Prompt:\n{formatted_sql_prompt}")

    # Example 4: Format another prompt
    print("\n[Example 4: Format 'error_correction_prompt' Template]")
    formatted_error_prompt = prompt_manager.format_prompt(
        template_name="error_correction_prompt",
        previous_sql="SELECT * FROM users WHERE email = test@example.com", # Missing quotes
        error_message="Syntax error near @test",
        db_schema=db_schema_example
    )
    # print(f"Formatted Error Correction Prompt:\n{formatted_error_prompt}")

    # Example 5: Load more templates (e.g., from a simulated file)
    # Simulate path to a JSON file containing more templates
    simulated_json_file_path = "additional_templates.json"
    # The dummy load_templates method has internal simulation for this path
    print("\n[Example 5: Load Additional Templates from Simulated JSON File]")
    prompt_manager.load_templates(simulated_json_file_path) # This will use the dummy file content
    prompt_manager.list_available_templates() # Show updated list

    # Try to format a newly "loaded" template
    if "greeting_template" in prompt_manager.list_available_templates(): # Check if dummy load worked
        print("\n[Example 5b: Format Newly Loaded 'greeting_template']")
        greeting = prompt_manager.format_prompt(
            template_name="greeting_template",
            name="Alex",
            place="the NovaSQLAgent examples"
        )
        # print(f"Formatted Greeting: {greeting}")
    else:
        print("\n'greeting_template' was not loaded from simulated file.")

    # Example 6: Attempt to format a non-existent template
    print("\n[Example 6: Attempt to Format Non-existent Template]")
    non_existent_prompt = prompt_manager.format_prompt(template_name="no_such_template_exists", data="test")
    # print(f"Result of formatting non-existent template: {non_existent_prompt}")


    print("\n--- PromptTemplateManager Module Example Complete ---")

if __name__ == "__main__":
    main()
