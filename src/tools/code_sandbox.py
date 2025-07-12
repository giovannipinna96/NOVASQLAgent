"""
This module implements a sandbox for code execution using the Code Sandbox MCP framework.
It provides a secure and isolated environment for running commands and scripts in a Docker container.
"""

import logging
import subprocess
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class CodeSandbox:
    """
    A sandbox for executing code in a Docker container using the Code Sandbox MCP framework.
    """

    def __init__(
        self,
        container_id: Optional[str] = None,
        image: str = "python:3.12-slim-bookworm",
        timeout: int = 60,
    ):
        """
        Initializes the CodeSandbox.

        Args:
            container_id: The ID of an existing container to connect to.
            image: The Docker image to use for the container.
            timeout: The timeout for command execution in seconds.
        """
        self.container_id = container_id
        self.image = image
        self.timeout = timeout

    def _run_mcp_command(self, command: List[str]) -> Dict[str, Any]:
        """
        Runs a command using the Code Sandbox MCP CLI.

        Args:
            command: The command to run.

        Returns:
            A dictionary containing the result of the command.
        """
        try:
            process = subprocess.run(
                ["code-sandbox-mcp"] + command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True,
            )
            return {
                "status": "success",
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "return_code": process.returncode,
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "stdout": e.stdout.strip(),
                "stderr": e.stderr.strip(),
                "return_code": e.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stderr": f"Command timed out after {self.timeout} seconds.",
            }
        except Exception as e:
            return {"status": "error", "stderr": str(e)}

    def initialize(self) -> Optional[str]:
        """
        Initializes a new sandbox environment.

        Returns:
            The ID of the new container, or None if initialization fails.
        """
        if self.container_id:
            logger.warning(f"Sandbox already initialized with container ID: {self.container_id}")
            return self.container_id

        result = self._run_mcp_command(["sandbox_initialize", "--image", self.image])
        if result["status"] == "success":
            self.container_id = result["stdout"]
            logger.info(f"Sandbox initialized with container ID: {self.container_id}")
            return self.container_id
        else:
            logger.error(f"Failed to initialize sandbox: {result['stderr']}")
            return None

    def copy_project(self, local_src_dir: Union[str, Path], dest_dir: str) -> bool:
        """
        Copies a directory to the sandbox.

        Args:
            local_src_dir: The path to the local directory.
            dest_dir: The destination path in the sandbox.

        Returns:
            True if the directory was copied successfully, False otherwise.
        """
        if not self.container_id:
            logger.error("Sandbox not initialized.")
            return False

        result = self._run_mcp_command(
            ["copy_project", "--container_id", self.container_id, "--local_src_dir", str(local_src_dir), "--dest_dir", dest_dir]
        )
        if result["status"] == "success":
            logger.info(f"Copied directory '{local_src_dir}' to '{dest_dir}' in container '{self.container_id}'")
            return True
        else:
            logger.error(f"Failed to copy directory: {result['stderr']}")
            return False

    def write_file(self, file_name: str, file_contents: str, dest_dir: Optional[str] = None) -> bool:
        """
        Writes a file to the sandbox.

        Args:
            file_name: The name of the file to create.
            file_contents: The contents of the file.
            dest_dir: The destination directory in the sandbox.

        Returns:
            True if the file was written successfully, False otherwise.
        """
        if not self.container_id:
            logger.error("Sandbox not initialized.")
            return False

        command = ["write_file", "--container_id", self.container_id, "--file_name", file_name, "--file_contents", file_contents]
        if dest_dir:
            command.extend(["--dest_dir", dest_dir])

        result = self._run_mcp_command(command)
        if result["status"] == "success":
            logger.info(f"Wrote file '{file_name}' to container '{self.container_id}'")
            return True
        else:
            logger.error(f"Failed to write file: {result['stderr']}")
            return False

    def execute(self, commands: List[str]) -> Dict[str, Any]:
        """
        Executes commands in the sandbox.

        Args:
            commands: A list of commands to execute.

        Returns:
            A dictionary containing the result of the execution.
        """
        if not self.container_id:
            return {"status": "error", "stderr": "Sandbox not initialized."}

        return self._run_mcp_command(["sandbox_exec", "--container_id", self.container_id] + ["--commands"] + commands)

    def copy_file(self, local_src_file: Union[str, Path], dest_path: str) -> bool:
        """
        Copies a file to the sandbox.

        Args:
            local_src_file: The path to the local file.
            dest_path: The destination path in the sandbox.

        Returns:
            True if the file was copied successfully, False otherwise.
        """
        if not self.container_id:
            logger.error("Sandbox not initialized.")
            return False

        result = self._run_mcp_command(
            ["copy_file", "--container_id", self.container_id, "--local_src_file", str(local_src_file), "--dest_path", dest_path]
        )
        if result["status"] == "success":
            logger.info(f"Copied file '{local_src_file}' to '{dest_path}' in container '{self.container_id}'")
            return True
        else:
            logger.error(f"Failed to copy file: {result['stderr']}")
            return False

    def stop(self) -> bool:
        """
        Stops and removes the sandbox container.

        Returns:
            True if the container was stopped successfully, False otherwise.
        """
        if not self.container_id:
            logger.error("Sandbox not initialized.")
            return False

        result = self._run_mcp_command(["sandbox_stop", "--container_id", self.container_id])
        if result["status"] == "success":
            logger.info(f"Stopped container '{self.container_id}'")
            self.container_id = None
            return True
        else:
            logger.error(f"Failed to stop container: {result['stderr']}")
            return False

    def get_logs(self) -> Optional[str]:
        """
        Gets the logs of the container.

        Returns:
            The logs of the container, or None if an error occurs.
        """
        if not self.container_id:
            logger.error("Sandbox not initialized.")
            return None

        result = self._run_mcp_command(["--resource", f"containers://{self.container_id}/logs"])
        if result["status"] == "success":
            return result["stdout"]
        else:
            logger.error(f"Failed to get container logs: {result['stderr']}")
            return None
