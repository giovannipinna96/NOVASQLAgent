"""
Module for a filesystem-based sandbox for executing arbitrary commands and Python scripts securely.
Focuses on restricting execution to a working directory and capturing output.
Resource and network restrictions are best-effort with standard Python libraries.
"""
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import os
import sys

# Attempt to import the 'resource' module for memory limits (Unix-specific)
try:
    import resource
except ImportError:
    resource = None # type: ignore
    logging.warning("FilesystemSandbox: 'resource' module not found. Memory limiting will not be available (this is expected on non-Unix systems like Windows).")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

class FilesystemSandbox:
    """
    A sandbox for executing filesystem commands (shell commands, Python scripts)
    with some safety controls like working directory restriction, timeout, and
    attempted memory limits (on Unix).
    """

    def __init__(
        self,
        working_dir: Union[str, Path],
        timeout_seconds: int = 10,
        memory_limit_mb: Optional[int] = None,
        allow_network: bool = False, # Conceptual flag, actual enforcement is limited
        python_executable: Optional[str] = None,
    ):
        """
        Initializes the FilesystemSandbox.

        Args:
            working_dir: The directory where commands will be executed.
                         It will be created if it doesn't exist.
            timeout_seconds: Maximum execution time for commands in seconds.
            memory_limit_mb: Optional memory limit in megabytes (Unix-only, best-effort).
            allow_network: A conceptual flag. If False, the sandbox aims to restrict
                           network access, though robust enforcement is hard with pure Python
                           subprocess without OS-level tools. This flag is more for intent.
            python_executable: Path to the Python interpreter to use for executing scripts.
                               Defaults to sys.executable.
        """
        self.working_dir = Path(working_dir).resolve()
        try:
            self.working_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create working directory {self.working_dir}: {e}")
            raise # Re-raise, as a working dir is critical.

        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.allow_network = allow_network # Store for potential future use or checks
        self.python_executable = python_executable or sys.executable

        logger.info(
            f"FilesystemSandbox initialized: working_dir='{self.working_dir}', "
            f"timeout={timeout_seconds}s, memory_limit_mb={memory_limit_mb or 'None'}, "
            f"allow_network={allow_network}"
        )
        if not self.allow_network:
            logger.warning(
                "allow_network=False is set, but robust network blocking via pure Python "
                "subprocess is limited. Use OS-level tools (containers, firewalls) for strict network isolation."
            )
        if self.memory_limit_mb and resource is None:
            logger.warning("Memory limit specified, but 'resource' module unavailable. Limit will not be enforced.")


    def _set_resource_limits(self) -> None:
        """
        Sets memory limits for the child process (Unix-only).
        This function is intended to be called by `subprocess.Popen` via `preexec_fn`.
        WARNING: Using `preexec_fn` is not safe if the parent process uses threads.
        For a library, it's better to run a wrapper script that sets limits then execs the target.
        However, for this self-contained class, we'll note the limitation.
        """
        if self.memory_limit_mb and resource:
            try:
                # Convert MB to bytes
                soft_limit = self.memory_limit_mb * 1024 * 1024
                hard_limit = soft_limit # Often set the same or slightly higher

                # RLIMIT_AS is the address space limit (virtual memory).
                # RLIMIT_DATA for data segment, RLIMIT_STACK for stack.
                # RLIMIT_AS is generally the most effective for overall memory.
                resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))
                # One could also set RLIMIT_DATA, RLIMIT_STACK if needed.
                # resource.setrlimit(resource.RLIMIT_DATA, (soft_limit, hard_limit))
                # resource.setrlimit(resource.RLIMIT_STACK, (soft_limit // 2, hard_limit // 2)) # Stack is usually smaller
                logger.debug(f"Process {os.getpid()}: Set memory limit to {self.memory_limit_mb}MB via RLIMIT_AS.")
            except Exception as e:
                # This error occurs in the child process, hard to propagate back directly.
                # Logging it here might not be visible if stdout/stderr of child are captured.
                # A more robust solution involves a dedicated wrapper.
                sys.stderr.write(f"FilesystemSandbox child process: Failed to set memory limit: {e}\n")
                # os._exit(127) # Exit child if limits can't be set, to prevent insecure execution.
                               # This is aggressive; for now, just log and continue.


    def execute_command(
        self,
        command: Union[str, List[str]],
        shell: bool = False,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Executes a given command in the sandbox.

        Args:
            command: The command to execute, as a string (if shell=True) or list of arguments.
            shell: If True, execute the command through the system's shell.
                   WARNING: shell=True can be a security hazard if `command` comes from
                   untrusted input. It's generally safer to use shell=False with a list of args.
                   This implementation defaults to False if command is a list.
            env: Optional dictionary of environment variables for the command.

        Returns:
            A dictionary containing execution results:
            {
                "status": "success" | "error" | "timeout" | "memory_exceeded",
                "output": Captured stdout string,
                "error_message": Captured stderr string or custom error message,
                "stack_trace": Optional Python stack trace if a Python script failed,
                "execution_time": Float seconds,
                "return_code": Integer return code of the command,
            }
        """
        if isinstance(command, str) and not shell:
            # If command is a string but shell=False, it needs to be split for subprocess.
            # However, robustly splitting a command string is complex (shlex.split).
            # Forcing users to provide a list for shell=False is safer.
            # Or, if a string is given with shell=False, assume it's a single executable path.
            # Let's assume `command` is a list if shell=False, or string if shell=True.
            # If string and shell=False, it's path to executable.
            pass # subprocess.run handles string command if it's just the executable.

        if isinstance(command, list) and shell:
            logger.warning("Executing a list of commands with shell=True. Consider joining them or using shell=False.")
            command = subprocess.list2cmdline(command) # Convert list to string for shell=True

        start_time = time.monotonic()
        result_status = "error" # Default status
        stdout_str = ""
        stderr_str = ""
        return_code = None
        stack_trace: Optional[str] = None

        # Safety: Ensure command is executed within the working directory.
        # `cwd` parameter in subprocess.run handles this.

        # `preexec_fn` is used for resource limits. Note its caveats with threads.
        preexec_fn_to_use = self._set_resource_limits if (self.memory_limit_mb and resource) else None

        try:
            process = subprocess.run(
                command,
                shell=shell,
                cwd=self.working_dir,
                capture_output=True,
                text=True, # Decodes stdout/stderr as text
                timeout=self.timeout_seconds,
                env=env, # Pass custom environment if provided
                preexec_fn=preexec_fn_to_use,
                check=False # Do not raise CalledProcessError, handle return code manually
            )
            stdout_str = process.stdout
            stderr_str = process.stderr
            return_code = process.returncode

            if process.returncode == 0:
                result_status = "success"
            else:
                # Check for potential out-of-memory kill signals (common return codes)
                # SIGKILL (9) or specific OOM killer signals might result in codes like 137 (128+9)
                # This is heuristic and OS-dependent.
                if resource and self.memory_limit_mb and (return_code == 137 or return_code == -9): # Killed by SIGKILL
                    result_status = "memory_exceeded"
                    stderr_str += "\nError: Process likely killed due to exceeding memory limits."
                else:
                    result_status = "error"
                logger.warning(f"Command execution failed with return code {return_code}. Stderr: {stderr_str[:200]}")

        except subprocess.TimeoutExpired:
            result_status = "timeout"
            stderr_str = f"Command timed out after {self.timeout_seconds} seconds."
            logger.warning(stderr_str)
            # return_code remains None or could be set to a specific timeout code if desired
        except FileNotFoundError:
            result_status = "error"
            stderr_str = f"Command not found: {str(command)[:100]}"
            logger.error(stderr_str)
        except Exception as e:
            result_status = "error"
            stderr_str = f"An unexpected error occurred during command execution: {str(e)}"
            # Try to capture stack trace if it's a Python-related error from this script itself
            import traceback
            stack_trace = traceback.format_exc()
            logger.error(stderr_str, exc_info=True)

        execution_time = time.monotonic() - start_time

        return {
            "status": result_status,
            "output": stdout_str.strip(),
            "error_message": stderr_str.strip(),
            "stack_trace": stack_trace,
            "execution_time": round(execution_time, 4),
            "return_code": return_code,
        }

    def execute_python_script(
        self,
        script_path: Union[str, Path], # Path relative to working_dir, or absolute
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Executes a Python script within the sandbox.

        Args:
            script_path: Path to the Python script. If relative, resolved against working_dir.
            args: Optional list of arguments to pass to the script.
            env: Optional dictionary of environment variables.

        Returns:
            A dictionary with execution results, similar to `execute_command`.
        """
        script_file = Path(script_path)
        if not script_file.is_absolute():
            script_file = self.working_dir / script_file

        if not script_file.exists():
            logger.error(f"Python script not found: {script_file}")
            return {
                "status": "error", "output": "", "error_message": f"Script not found: {script_file}",
                "stack_trace": None, "execution_time": 0.0, "return_code": -1, # Custom error code
            }

        command = [self.python_executable, str(script_file)]
        if args:
            command.extend(args)

        logger.info(f"Executing Python script: {' '.join(command)} in {self.working_dir}")
        return self.execute_command(command, shell=False, env=env)

    def create_file(self, relative_path: Union[str, Path], content: str) -> bool:
        """Creates a file with given content within the sandbox's working directory."""
        try:
            file_path = self.working_dir / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dirs exist
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"File created: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create file {relative_path} in sandbox: {e}")
            return False

    def read_file(self, relative_path: Union[str, Path]) -> Optional[str]:
        """Reads a file from the sandbox's working directory."""
        try:
            file_path = self.working_dir / relative_path
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"File not found or is not a file: {file_path}")
                return None
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to read file {relative_path} from sandbox: {e}")
            return None

    def list_files(self, relative_path: Union[str, Path] = ".") -> Optional[List[str]]:
        """Lists files and directories within a given path relative to the working directory."""
        try:
            target_path = (self.working_dir / relative_path).resolve()
            # Security check: ensure target_path is still within self.working_dir
            if self.working_dir not in target_path.parents and target_path != self.working_dir:
                 logger.error(f"Attempt to list files outside sandbox working directory: {target_path}")
                 return None # Or raise error

            if not target_path.exists() or not target_path.is_dir():
                logger.warning(f"Path not found or not a directory for listing: {target_path}")
                return None
            return [item.name for item in target_path.iterdir()]
        except Exception as e:
            logger.error(f"Failed to list files in {relative_path} within sandbox: {e}")
            return None


if __name__ == "__main__":
    # This block is for illustrative purposes and basic structural checks.
    # Actual execution and thorough testing require a proper environment.
    logger.info("FilesystemSandbox: Illustrative __main__ block.")

    # Create a temporary directory for the sandbox to operate in for this example
    with tempfile.TemporaryDirectory(prefix="fs_sandbox_test_") as tmpdir_name:
        sandbox_path = Path(tmpdir_name) / "my_sandbox_work"
        logger.info(f"Illustrative sandbox working directory: {sandbox_path}")

        try:
            # Initialize sandbox
            # Memory limit test might only work on Unix and if script has permissions
            fs_sandbox = FilesystemSandbox(sandbox_path, timeout_seconds=5, memory_limit_mb=64 if resource else None)

            # Example 1: Simple echo command
            logger.info("\n--- Example 1: Echo command ---")
            # On Windows, 'echo' is a shell built-in. On Linux, it's usually /bin/echo.
            # For cross-platform, using a list is better if not using shell=True.
            # However, `echo` itself is tricky. `sys.executable -c "print('Hello')"` is more reliable.
            # For illustration, let's use a Python print command.
            py_echo_cmd = [fs_sandbox.python_executable, "-c", "import sys; sys.stdout.write('Hello Sandbox stdout'); sys.stderr.write('Hello Sandbox stderr')"]
            result_echo = fs_sandbox.execute_command(py_echo_cmd)
            print(f"Echo Result: {result_echo}")
            assert result_echo["status"] == "success"
            assert "Hello Sandbox stdout" in result_echo["output"]
            assert "Hello Sandbox stderr" in result_echo["error_message"] # Stderr goes to error_message

            # Example 2: Create and execute a Python script
            logger.info("\n--- Example 2: Python script execution ---")
            script_content = "import sys\nprint(f'Hello from script! Args: {sys.argv[1:]}')\n"
            fs_sandbox.create_file("test_script.py", script_content)

            result_script = fs_sandbox.execute_python_script("test_script.py", args=["arg1", "val1"])
            print(f"Python Script Result: {result_script}")
            assert result_script["status"] == "success"
            assert "Hello from script!" in result_script["output"]
            assert "['arg1', 'val1']" in result_script["output"]

            # Example 3: Timeout
            logger.info("\n--- Example 3: Timeout test ---")
            # Command that sleeps longer than timeout
            # Windows: "timeout /t 10 /nobreak > NUL" or ping trick: "ping -n 10 127.0.0.1 > NUL"
            # Linux: "sleep 10"
            sleep_cmd = [fs_sandbox.python_executable, "-c", "import time; time.sleep(10)"]
            timeout_sandbox = FilesystemSandbox(sandbox_path, timeout_seconds=2)
            result_timeout = timeout_sandbox.execute_command(sleep_cmd)
            print(f"Timeout Result: {result_timeout}")
            assert result_timeout["status"] == "timeout"

            # Example 4: File operations
            logger.info("\n--- Example 4: File operations ---")
            fs_sandbox.create_file("example.txt", "File content here.")
            read_content = fs_sandbox.read_file("example.txt")
            print(f"Read content: {read_content}")
            assert read_content == "File content here."
            file_list = fs_sandbox.list_files(".")
            print(f"Files in sandbox: {file_list}")
            assert "example.txt" in file_list if file_list else False
            assert "test_script.py" in file_list if file_list else False

            # Example 5: Command error
            logger.info("\n--- Example 5: Command error ---")
            error_cmd = [fs_sandbox.python_executable, "-c", "import sys; sys.stderr.write('Error output'); sys.exit(5)"]
            result_error_cmd = fs_sandbox.execute_command(error_cmd)
            print(f"Command Error Result: {result_error_cmd}")
            assert result_error_cmd["status"] == "error"
            assert result_error_cmd["return_code"] == 5
            assert "Error output" in result_error_cmd["error_message"]


        except Exception as e:
            logger.error(f"Error in FilesystemSandbox illustrative __main__: {e}", exc_info=True)
        finally:
            logger.info(f"Illustrative __main__ finished. Sandbox contents were in {sandbox_path}")
            # The tempdir_name will be cleaned up automatically by TemporaryDirectory context manager

    logger.info("FilesystemSandbox illustrative __main__ block completed.")
