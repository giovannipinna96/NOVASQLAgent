"""
Tests for src/model/prompt_template_manager.py
As per instructions, this file will only contain the code, without attempting to run it
or install dependencies like 'pytest'.
"""
import unittest
from pathlib import Path
import tempfile
import sys
import io # For capturing print output

# Add src directory to sys.path to allow import of prompt_template_manager
# This is a common way to handle imports in tests when the module is not installed
# Adjust the number of .parent calls if the test structure is different
try:
    SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from model.prompt_template_manager import PromptTemplateManager
except ImportError as e:
    # This allows the file to be parsed even if the module or its deps are not found,
    # which is relevant given the "do not install/run" constraint.
    print(f"Test_PromptTemplateManager: Could not import PromptTemplateManager. Error: {e}")
    PromptTemplateManager = None # type: ignore

# Mock unittest.mock if it's not available in a very minimal Python environment
# This is unlikely for standard Python but added for extreme robustness in non-execution context.
if not hasattr(unittest, 'mock'):
    try:
        from unittest import mock
        unittest.mock = mock
    except ImportError:
        print("Test_PromptTemplateManager: unittest.mock not found. Some tests for display methods might not be fully structured.")
        # Create a dummy mock patch decorator if unavailable
        class DummyMockPatch:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, func):
                def wrapper(*args, **kwargs):
                    print(f"Warning: unittest.mock.patch called for {func.__name__} but mock is unavailable. Stdout not captured.")
                    return func(*args, **kwargs) # Call original function without mocking stdout
                return wrapper
            def start(self): pass
            def stop(self): pass
        unittest.mock = type('MockModule', (object,), {'patch': DummyMockPatch})() # type: ignore



class TestPromptTemplateManager(unittest.TestCase):
    """
    Unit tests for the PromptTemplateManager class.
    """

    def setUp(self):
        """
        Set up a temporary directory and dummy template files for testing.
        """
        if PromptTemplateManager is None:
            self.skipTest("PromptTemplateManager module not loaded, skipping tests.")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.valid_template_content = "Hello, {name}! Today is {day}."
        self.valid_template_path = self.temp_path / "valid_template.txt"
        with open(self.valid_template_path, "w", encoding="utf-8") as f:
            f.write(self.valid_template_content)

        self.no_placeholder_template_content = "This is a static template."
        self.no_placeholder_template_path = self.temp_path / "no_placeholder_template.txt"
        with open(self.no_placeholder_template_path, "w", encoding="utf-8") as f:
            f.write(self.no_placeholder_template_content)

        self.empty_template_content = ""
        self.empty_template_path = self.temp_path / "empty_template.txt"
        with open(self.empty_template_path, "w", encoding="utf-8") as f:
            f.write(self.empty_template_content)

        self.duplicate_template_content = "Name: {name}, Age: {age}, Name again: {name}."
        self.duplicate_template_path = self.temp_path / "duplicate_template.txt"
        with open(self.duplicate_template_path, "w", encoding="utf-8") as f:
            f.write(self.duplicate_template_content)

    def tearDown(self):
        """ Clean up the temporary directory after tests. """
        self.temp_dir.cleanup()

    def test_initialization_valid_file(self):
        """Test successful initialization."""
        manager = PromptTemplateManager(self.valid_template_path)
        self.assertEqual(manager.raw_template, self.valid_template_content)
        self.assertListEqual(sorted(manager.expected_keys), sorted(["name", "day"]))

    def test_initialization_file_not_found(self):
        """Test FileNotFoundError for non-existent template."""
        with self.assertRaises(FileNotFoundError):
            PromptTemplateManager(self.temp_path / "non_existent.txt")

    def test_initialization_invalid_extension(self):
        """Test ValueError for non .txt file."""
        invalid_ext_path = self.temp_path / "template.md"
        with open(invalid_ext_path, "w") as f: f.write("md")
        with self.assertRaisesRegex(ValueError, "Template file must be a .txt file."):
            PromptTemplateManager(invalid_ext_path)

    def test_format_prompt_valid_inputs(self):
        """Test successful prompt formatting."""
        manager = PromptTemplateManager(self.valid_template_path)
        inputs = {"name": "Alice", "day": "Wednesday"}
        expected = "Hello, Alice! Today is Wednesday."
        self.assertEqual(manager.format_prompt(inputs), expected)

    def test_format_prompt_missing_key(self):
        """Test KeyError for missing formatting key."""
        manager = PromptTemplateManager(self.valid_template_path)
        with self.assertRaisesRegex(KeyError, "Missing keys for formatting: day"):
            manager.format_prompt({"name": "Bob"})

    def test_format_prompt_extra_keys(self):
        """Test formatting ignores extra keys."""
        manager = PromptTemplateManager(self.valid_template_path)
        inputs = {"name": "Charlie", "day": "Friday", "mood": "happy"}
        expected = "Hello, Charlie! Today is Friday."
        self.assertEqual(manager.format_prompt(inputs), expected)

    def test_format_prompt_no_placeholders(self):
        """Test formatting for template with no placeholders."""
        manager = PromptTemplateManager(self.no_placeholder_template_path)
        self.assertEqual(manager.format_prompt({"name": "Dana"}), self.no_placeholder_template_content)
        self.assertEqual(manager.format_prompt({}), self.no_placeholder_template_content)

    def test_format_prompt_empty_template(self):
        """Test formatting for an empty template."""
        manager = PromptTemplateManager(self.empty_template_path)
        self.assertEqual(manager.format_prompt({"key": "value"}), "")

    def test_expected_keys_property(self):
        """Test expected_keys property behavior."""
        manager_valid = PromptTemplateManager(self.valid_template_path)
        self.assertListEqual(sorted(manager_valid.expected_keys), sorted(["day", "name"]))

        manager_no_placeholders = PromptTemplateManager(self.no_placeholder_template_path)
        self.assertListEqual(manager_no_placeholders.expected_keys, [])

        manager_duplicate = PromptTemplateManager(self.duplicate_template_path)
        # Regex findall preserves order of first encounters, list(dict.fromkeys()) also preserves order.
        # The implementation uses a method to ensure uniqueness while preserving first-encounter order.
        self.assertListEqual(manager_duplicate.expected_keys, ["name", "age"])

    def test_raw_template_property(self):
        """Test raw_template property."""
        manager = PromptTemplateManager(self.valid_template_path)
        self.assertEqual(manager.raw_template, self.valid_template_content)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO) # type: ignore
    def test_display_template_overview(self, mock_stdout):
        """Test display_template_overview prints expected information."""
        manager = PromptTemplateManager(self.valid_template_path)
        manager.display_template_overview()
        output = mock_stdout.getvalue()
        self.assertIn("--- Prompt Template Overview ---", output)
        self.assertIn(f"Template File: {self.valid_template_path.name}", output)
        self.assertIn("Raw Template Structure:", output)
        self.assertIn(self.valid_template_content, output)
        self.assertIn("Expected Keys for Formatting:", output)
        self.assertIn("- {name}", output) # Using actual key names
        self.assertIn("- {day}", output)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO) # type: ignore
    def test_display_rendered_prompt_valid(self, mock_stdout):
        """Test display_rendered_prompt with valid inputs."""
        manager = PromptTemplateManager(self.valid_template_path)
        inputs = {"name": "Alice", "day": "Wednesday"}
        expected_rendered = "Hello, Alice! Today is Wednesday."
        manager.display_rendered_prompt(inputs)
        output = mock_stdout.getvalue()
        self.assertIn("--- Rendered Prompt ---", output)
        self.assertIn(expected_rendered, output)
        self.assertIn("--- End of Rendered Prompt ---", output)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO) # type: ignore
    def test_display_rendered_prompt_missing_key(self, mock_stdout):
        """Test display_rendered_prompt handles and prints error for missing key."""
        manager = PromptTemplateManager(self.valid_template_path)
        inputs = {"name": "Bob"} # Missing "day"
        manager.display_rendered_prompt(inputs)
        output = mock_stdout.getvalue()
        # The internal logger in PromptTemplateManager might also print.
        # The method itself prints "Could not render prompt: ..."
        self.assertIn("Could not render prompt: Missing keys for formatting: day", output)
        self.assertNotIn("--- Rendered Prompt ---", output) # This header shouldn't appear on error


if __name__ == "__main__":
    if PromptTemplateManager is not None and hasattr(unittest, 'mock'):
        print("Running PromptTemplateManager tests (illustrative execution)...")
        unittest.main(verbosity=2)
    else:
        reason = "PromptTemplateManager module not loaded"
        if not hasattr(unittest, 'mock'): reason += " or unittest.mock not available"
        print(f"Skipping PromptTemplateManager tests: {reason}.")
        print("This is expected if dependencies are not installed or src path is incorrect in a non-execution environment.")
