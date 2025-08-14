"""HuggingFace Agent with enhanced tool capabilities.

This module implements a sophisticated AI agent using smolagents framework with:
- Comprehensive bash command execution tools
- File system operations with safety checks
- Mathematical computation tools
- Type-safe interfaces with error handling
- Factory pattern for model loading
- Command pattern for tool execution

Design Patterns:
- Factory Pattern: Model and tool creation
- Command Pattern: Tool execution abstraction
- Strategy Pattern: Different tool execution strategies
- Template Method: Common tool operation patterns
"""

import argparse
import subprocess
import os
from pathlib import Path
from typing import Optional, List, Any, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from smolagents import tool
from smolagents import CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result container for tool execution with status tracking."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: Optional[float] = None


class BaseToolExecutor(ABC):
    """Abstract base class for tool executors using Template Method pattern."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> ToolExecutionResult:
        """Execute the tool operation."""
        pass
    
    def validate_input(self, *args, **kwargs) -> bool:
        """Validate input parameters before execution."""
        return True
    
    def post_process(self, result: str) -> str:
        """Post-process the execution result."""
        return result.strip() if result else "No output"


class BashToolExecutor(BaseToolExecutor):
    """Executor for bash command operations with enhanced security."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.forbidden_commands = [
            'rm -rf /', 'format', 'del *', 'sudo rm',
            'dd if=', ':(){ :|:& };:', 'chmod -R 777'
        ]
    
    def validate_input(self, command: str) -> bool:
        """Validate command for security risks."""
        if not command or not command.strip():
            return False
        
        command_lower = command.lower()
        for forbidden in self.forbidden_commands:
            if forbidden in command_lower:
                logger.warning(f"Forbidden command detected: {forbidden}")
                return False
        
        return True
    
    def execute(self, command: str) -> ToolExecutionResult:
        """Execute bash command with security validation."""
        if not self.validate_input(command):
            return ToolExecutionResult(
                success=False,
                output="",
                error="Invalid or forbidden command"
            )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                output = self.post_process(result.stdout)
                return ToolExecutionResult(
                    success=True,
                    output=output or "Command executed successfully"
                )
            else:
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                return ToolExecutionResult(
                    success=False,
                    output="",
                    error=f"Error (code {result.returncode}): {error_msg}"
                )
        
        except subprocess.TimeoutExpired:
            return ToolExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}"
            )


class FileSystemToolExecutor(BaseToolExecutor):
    """Executor for file system operations with path validation."""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):  # 10MB
        self.max_file_size = max_file_size
    
    def validate_path(self, path: str) -> bool:
        """Validate file path for security."""
        try:
            path_obj = Path(path).resolve()
            # Prevent directory traversal attacks
            if '..' in str(path_obj) or path_obj.is_absolute() and not str(path_obj).startswith(str(Path.cwd())):
                return False
            return True
        except Exception:
            return False
    
    def list_directory(self, path: str = ".") -> ToolExecutionResult:
        """List directory contents safely."""
        if not self.validate_path(path):
            return ToolExecutionResult(
                success=False,
                output="",
                error="Invalid or unsafe path"
            )
        
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return ToolExecutionResult(
                    success=False,
                    output="",
                    error=f"Path '{path}' does not exist"
                )
            
            if not path_obj.is_dir():
                return ToolExecutionResult(
                    success=False,
                    output="",
                    error=f"Path '{path}' is not a directory"
                )
            
            items = []
            for item in sorted(path_obj.iterdir()):
                item_type = "DIR" if item.is_dir() else "FILE"
                items.append(f"{item_type}: {item.name}")
            
            output = "\n".join(items) if items else "Directory is empty"
            return ToolExecutionResult(success=True, output=output)
            
        except Exception as e:
            return ToolExecutionResult(
                success=False,
                output="",
                error=f"Error listing directory: {str(e)}"
            )


# Initialize tool executors
bash_executor = BashToolExecutor()
fs_executor = FileSystemToolExecutor()


# === ENHANCED TOOLS WITH DESIGN PATTERNS ===
@tool
def execute_bash_command(command: str) -> str:
    """
    Execute a bash command with enhanced security and error handling.
    
    Uses Command Pattern through BashToolExecutor for safe execution
    with built-in security validations and timeout protection.
    
    Args:
        command: The bash command to execute (e.g., "ls -la", "pwd", "echo 'hello'")
    
    Returns:
        str: The command output or detailed error message
    """
    result = bash_executor.execute(command)
    
    if result.success:
        return result.output
    else:
        return result.error or "Unknown error occurred"


@tool  
def list_directory(path: str = ".") -> str:
    """
    List files and directories with enhanced security validation.
    
    Uses FileSystemToolExecutor with path validation to prevent
    directory traversal attacks and ensure safe file operations.
    
    Args:
        path: The directory path to list (defaults to current directory)
    
    Returns:
        str: List of files and directories or error message
    """
    result = fs_executor.list_directory(path)
    
    if result.success:
        return result.output
    else:
        return result.error or "Failed to list directory"


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
