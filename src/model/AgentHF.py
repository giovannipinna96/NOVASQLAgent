import argparse
import subprocess
import os
from pathlib import Path
from smolagents import tool
from smolagents import CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# === TOOLS ===ù
@tool
def add_numbers(a: float, b: float) -> float:
    """
    Sum of two numbers. Returns the sum of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers. Returns the product of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a * b

@tool
def devide_numbers(a: float, b: float) -> float:
    """
    Devide two numbers. Returns the division of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a / b

@tool
def giovannetor(a: float, b: float) -> float:
    """
    The 'giovannetor' function give is calculated as (a ** b) + 1.

    Args:
      a: The base number.
      b: The exponent.
    """
    return a ** b + 1


@tool
def execute_bash_command(command: str) -> str:
    """
    Execute a bash command and return its output. Use this to run shell commands, system operations, file management, etc.
    
    Args:
        command: The bash command to execute (e.g., "ls -la", "pwd", "echo 'hello'", "mkdir test_dir")
    
    Returns:
        str: The command output (stdout) or error message if the command fails
    """
    try:
        # Execute command with shell=True for cross-platform compatibility
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30  # 30 second timeout for safety
        )
        
        if result.returncode == 0:
            return result.stdout.strip() if result.stdout else "Command executed successfully (no output)"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Command failed with no error message"
            return f"Error (return code {result.returncode}): {error_msg}"
            
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool  
def list_directory(path: str = ".") -> str:
    """
    List files and directories in a specific path. Useful for exploring the file system.
    
    Args:
        path: The directory path to list (defaults to current directory)
    
    Returns:
        str: List of files and directories in the specified path
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return f"Error: Path '{path}' does not exist"
        
        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"
        
        items = []
        for item in sorted(path_obj.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"{item_type}: {item.name}")
        
        return "\n".join(items) if items else "Directory is empty"
        
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def get_current_directory() -> str:
    """
    Get the current working directory path.
    
    Returns:
        str: The current working directory path
    """
    try:
        return str(Path.cwd())
    except Exception as e:
        return f"Error getting current directory: {str(e)}"


@tool
def change_directory(path: str) -> str:
    """
    Change the current working directory. Note: This only affects the agent's working context.
    
    Args:
        path: The directory path to change to
    
    Returns:
        str: Success message or error if the operation fails
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return f"Error: Directory '{path}' does not exist"
        
        if not path_obj.is_dir():
            return f"Error: Path '{path}' is not a directory"
        
        os.chdir(path)
        return f"Changed directory to: {Path.cwd()}"
        
    except Exception as e:
        return f"Error changing directory: {str(e)}"


@tool
def create_file(filename: str, content: str = "") -> str:
    """
    Create a new file with optional content.
    
    Args:
        filename: Name of the file to create
        content: Content to write to the file (optional)
    
    Returns:
        str: Success message or error if the operation fails
    """
    try:
        file_path = Path(filename)
        file_path.write_text(content, encoding='utf-8')
        return f"File '{filename}' created successfully"
        
    except Exception as e:
        return f"Error creating file: {str(e)}"


@tool
def read_file(filename: str) -> str:
    """
    Read the content of a file.
    
    Args:
        filename: Name of the file to read
    
    Returns:
        str: File content or error message if the operation fails
    """
    try:
        file_path = Path(filename)
        if not file_path.exists():
            return f"Error: File '{filename}' does not exist"
        
        return file_path.read_text(encoding='utf-8')
        
    except Exception as e:
        return f"Error reading file: {str(e)}"


# tools = [
#     Tool(name="add_numbers", description="Add two numbers", func=add_numbers),
#     Tool(name="multiply_numbers", description="Multiply two numbers", func=multiply_numbers),
#     Tool(name="giovannetor", description="Raise a to the power of b", func=giovannetor),
# ]


# === BUILD LLM ===
def build_llm(model_id: str, local_model_path: str = None):
    if local_model_path:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    return pipe


# === MAIN ===
def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Agent with bash command capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python AgentHF.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/local/model" --query "List the files in the current directory"
  python AgentHF.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/local/model" --query "Create a file called test.txt and write 'Hello World' in it"
  python AgentHF.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/local/model" --query "Run the command 'python --version' to check the Python version"
  python AgentHF.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/local/model" --query "Calculate 5 * 3 and then create a directory called result_15"
        
Available tools:
  - Math tools: add_numbers, multiply_numbers, devide_numbers, giovannetor
  - Bash tools: execute_bash_command, list_directory, get_current_directory, change_directory, create_file, read_file
        """
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--query", type=str, required=True, help="Query or task for the agent to execute")
    parser.add_argument("--local_model_path", type=str, default=None, help="Path to local model (overrides --model if provided)")
    args = parser.parse_args()

    # Load LLM pipeline
    # llm_pipeline = build_llm(args.model, args.local_model_path)
    model = TransformersModel(model_id=args.local_model_path)
    
    # Create agent with both math tools and bash command tools
    all_tools = [
        # Math tools
        add_numbers, multiply_numbers, giovannetor, devide_numbers,
        # Bash command tools
        execute_bash_command, list_directory, get_current_directory, 
        change_directory, create_file, read_file
    ]
    
    agent = CodeAgent(tools=all_tools, model=model)

    # Run the agent
    result = agent.run(args.query)

    print(f"\n========\nFinal Answer: {result}\n========")


if __name__ == "__main__":
    main()
