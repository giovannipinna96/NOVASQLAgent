"""
Module for orchestrating sandbox environments (Filesystem and SQL).
Manages configurations and delegates execution to the appropriate sandbox.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Attempt to import sandbox types.
# In a "no run" environment, these might be None if their files haven't been "processed"
# or if there were import errors within them (though they are designed to be parseable).
try:
    from .filesystem_sandbox import FilesystemSandbox
except ImportError:
    FilesystemSandbox = None # type: ignore
    logging.warning("SandboxExecutor: FilesystemSandbox not found or importable.")
try:
    from .sql_sandbox import SQLSandbox
except ImportError:
    SQLSandbox = None # type: ignore
    logging.warning("SandboxExecutor: SQLSandbox not found or importable.")

# For determining PROJECT_ROOT to find the top-level 'config' directory
import sys
import time # Already imported below, but good to have near related logic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

DEFAULT_SANDBOX_CONFIG_SUBDIR = "sandbox" # Subdirectory within the main 'config' folder

class SandboxExecutor:
    """
    Orchestrates the execution of tasks in different sandbox environments
    based on loaded configurations or direct parameters.
    """

    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        """
        Initializes the SandboxExecutor.

        Args:
            base_config_path: Path to the directory containing sandbox configuration files
                              (e.g., 'project_root/config/sandbox').
                              If None, defaults to 'PROJECT_ROOT/config/sandbox/'.
        """
        if base_config_path:
            self.config_dir = Path(base_config_path)
        else:
            try:
                # Assuming this file is in src/tools/sandbox_executor.py
                # PROJECT_ROOT would be two levels up from this file's parent.
                project_root = Path(__file__).resolve().parent.parent.parent
                self.config_dir = project_root / "config" / DEFAULT_SANDBOX_CONFIG_SUBDIR
            except NameError: # __file__ not defined
                # Fallback: current_working_directory/config/sandbox
                self.config_dir = Path.cwd() / "config" / DEFAULT_SANDBOX_CONFIG_SUBDIR
                logger.warning(
                    f"__file__ not defined, defaulting config_dir to: {self.config_dir}. "
                    "Ensure this path is correct or provide base_config_path."
                )

        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"SandboxExecutor initialized. Configuration directory: {self.config_dir}")
        except Exception as e:
            logger.error(f"Failed to create or access config directory {self.config_dir}: {e}")
            # Depending on strictness, might raise error or proceed allowing only ad-hoc configs.
            # For now, log and proceed. Saving/loading configs might fail.

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Loads a sandbox configuration from a JSON file.
        Assumes config files are named `config_name.json`.

        Args:
            config_name: The name of the configuration to load (without .json extension).

        Returns:
            A dictionary with the configuration data, or None if loading fails.
        """
        config_file = self.config_dir / f"{config_name}.json"
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_file}")
            return None
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            logger.info(f"Configuration '{config_name}' loaded from {config_file}")
            return config_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from configuration file {config_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load configuration '{config_name}': {e}")
            return None

    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Saves a sandbox configuration to a JSON file.
        Files will be named `config_name.json`.

        Args:
            config_name: The name for the configuration (without .json extension).
            config_data: The configuration dictionary to save.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not self.config_dir.exists():
            try: # Try to create it again if it failed during init but user proceeds
                self.config_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot save config: Config directory {self.config_dir} does not exist and cannot be created: {e}")
                return False

        config_file = self.config_dir / f"{config_name}.json"
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Configuration '{config_name}' saved to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration '{config_name}' to {config_file}: {e}")
            return False

    def run_in_sandbox(
        self,
        task_type: str, # "filesystem" or "sql"
        task_details: Dict[str, Any], # Contains command/query and specific args
        sandbox_config: Optional[Dict[str, Any]] = None, # Full sandbox config
        config_name: Optional[str] = None, # Name of a saved config to load
    ) -> Dict[str, Any]:
        """
        Runs a task in the specified sandbox environment.

        The configuration for the sandbox can be provided directly via `sandbox_config`,
        loaded by `config_name`, or defaults will be used.

        Args:
            task_type: Type of sandbox to use ("filesystem" or "sql").
            task_details: Dictionary containing task-specific information.
                For "filesystem": {"command": "ls -l", "shell": False, "env": None, "script_path": "path/to/script.py", "args": []}
                                  (either "command" or "script_path" should be primary)
                For "sql": {"query": "SELECT * FROM users", "params": None, "script": False}
            sandbox_config: An optional dictionary providing the full sandbox configuration.
                            If provided, `config_name` is ignored.
            config_name: Optional name of a pre-saved configuration file to use.

        Returns:
            A dictionary with the execution result from the sandbox.
            Returns an error result if configuration or sandbox type is invalid.
        """
        loaded_config = {}
        if sandbox_config:
            loaded_config = sandbox_config
            logger.info("Using provided sandbox_config dictionary.")
        elif config_name:
            loaded_config = self.load_config(config_name) or {} # Use empty if load fails, defaults will apply
            if not loaded_config and config_name: # Explicitly check if load_config returned None for a given name
                 logger.error(f"Failed to load specified config '{config_name}'. Proceeding with defaults for sandbox type '{task_type}'.")

        # Extract common sandbox parameters from loaded_config or use defaults
        # These defaults are conceptual; actual sandbox classes have their own defaults.
        # The config would override these.
        timeout = loaded_config.get("timeout_seconds", 10)
        memory_limit = loaded_config.get("memory_limit_mb", None) # Filesystem specific

        # Common result structure for initial errors
        error_result = lambda msg: {
            "status": "error", "output": None, "error_message": msg,
            "execution_time": 0.0
        }

        start_time_overall = time.monotonic()

        if task_type == "filesystem":
            if not FilesystemSandbox:
                return error_result("FilesystemSandbox module is not available.")

            working_dir = loaded_config.get("working_dir", f"./sandbox_fs_run_{int(time.time())}")
            allow_network = loaded_config.get("allow_network", False) # Filesystem specific
            python_exec = loaded_config.get("python_executable", sys.executable)

            try:
                fs_sandbox = FilesystemSandbox(
                    working_dir=Path(working_dir), # Ensure it's a Path
                    timeout_seconds=timeout,
                    memory_limit_mb=memory_limit,
                    allow_network=allow_network,
                    python_executable=python_exec
                )
            except Exception as e:
                return error_result(f"Failed to initialize FilesystemSandbox: {e}")

            command_to_run = task_details.get("command")
            script_to_run = task_details.get("script_path")

            if script_to_run:
                script_args = task_details.get("args")
                env_vars = task_details.get("env")
                return fs_sandbox.execute_python_script(script_to_run, args=script_args, env=env_vars)
            elif command_to_run:
                use_shell = task_details.get("shell", False if isinstance(command_to_run, list) else True if isinstance(command_to_run, str) else False) # Default shell based on command type
                env_vars = task_details.get("env")
                return fs_sandbox.execute_command(command_to_run, shell=use_shell, env=env_vars)
            else:
                return error_result("Filesystem task details missing 'command' or 'script_path'.")

        elif task_type == "sql":
            if not SQLSandbox:
                return error_result("SQLSandbox module is not available.")

            db_path = loaded_config.get("db_path", ":memory:")
            table_prefix = loaded_config.get("table_prefix", "sandbox_tbl_")
            enforce_safety = loaded_config.get("enforce_safety_checks", True)

            try:
                sql_sb = SQLSandbox(
                    db_path=db_path,
                    timeout_seconds=timeout, # SQLSandbox timeout is conceptual for its methods
                    table_prefix=table_prefix,
                    enforce_safety_checks=enforce_safety
                )
            except Exception as e:
                return error_result(f"Failed to initialize SQLSandbox: {e}")

            query = task_details.get("query")
            if not query:
                return error_result("SQL task details missing 'query'.")

            params = task_details.get("params")
            is_script = task_details.get("script", False)
            return sql_sb.execute_query(query, params=params, script=is_script)

        else:
            exec_time = time.monotonic() - start_time_overall
            return {
                "status": "error", "output": None,
                "error_message": f"Unsupported sandbox task_type: '{task_type}'. Must be 'filesystem' or 'sql'.",
                "execution_time": round(exec_time, 4)
            }


if __name__ == "__main__":
    logger.info("SandboxExecutor: Illustrative __main__ block.")
    # This assumes FilesystemSandbox and SQLSandbox are available (even if mocked)
    # and that their __init__ and execute methods can be called.

    # Setup a temporary config directory for this illustration
    with tempfile.TemporaryDirectory(prefix="sb_exec_configs_") as tmp_config_dir_name:
        config_dir = Path(tmp_config_dir_name)
        logger.info(f"Illustrative config directory: {config_dir}")

        executor = SandboxExecutor(base_config_path=config_dir)

        # 1. Create and save a filesystem sandbox configuration
        fs_config_data = {
            "sandbox_type": "filesystem", # Informational, executor uses task_type param
            "working_dir": str(Path(tmp_config_dir_name) / "fs_work"), # Ensure it's string for JSON
            "timeout_seconds": 5,
            "memory_limit_mb": 128,
            "allow_network": False,
            "python_executable": sys.executable
        }
        executor.save_config("default_fs", fs_config_data)

        # 2. Create and save an SQL sandbox configuration
        sql_config_data = {
            "sandbox_type": "sql",
            "db_path": str(Path(tmp_config_dir_name) / "sandbox_db.sqlite"),
            "timeout_seconds": 3,
            "table_prefix": "test_exec_",
            "enforce_safety_checks": True
        }
        executor.save_config("default_sql", sql_config_data)

        # 3. Run a filesystem task using a loaded configuration
        logger.info("\n--- Running filesystem task with loaded config ---")
        if FilesystemSandbox: # Check if mock or real class is available
            fs_task = {
                "command": [sys.executable, "-c", "import os; print(f'FS Sandbox says hello from {os.getcwd()}')"]
            }
            fs_result = executor.run_in_sandbox(task_type="filesystem", task_details=fs_task, config_name="default_fs")
            print(f"Filesystem task result (config): {fs_result}")
            # assert fs_result["status"] == "success" # Depends on mock behavior
            # assert "FS Sandbox says hello" in fs_result.get("output", "")
        else:
            logger.warning("Skipping filesystem task illustration as FilesystemSandbox is not available.")

        # 4. Run an SQL task using a loaded configuration
        logger.info("\n--- Running SQL task with loaded config ---")
        if SQLSandbox: # Check if mock or real class is available
            sql_task = {
                "query": f"CREATE TABLE IF NOT EXISTS {sql_config_data['table_prefix']}t1 (id INT); INSERT INTO {sql_config_data['table_prefix']}t1 VALUES (1); SELECT * FROM {sql_config_data['table_prefix']}t1;",
                "script": True # Execute as a script
            }
            sql_result = executor.run_in_sandbox(task_type="sql", task_details=sql_task, config_name="default_sql")
            print(f"SQL task result (config): {sql_result}")
            # assert sql_result["status"] == "success" # Depends on mock SQLite behavior
            # The SELECT output part of script won't be in sql_result["output"] with current SQLSandbox script exec.
        else:
            logger.warning("Skipping SQL task illustration as SQLSandbox is not available.")

        # 5. Run a filesystem task with ad-hoc (direct) configuration
        logger.info("\n--- Running filesystem task with ad-hoc config ---")
        if FilesystemSandbox:
            adhoc_fs_config = {
                "working_dir": str(Path(tmp_config_dir_name) / "fs_adhoc_work"),
                "timeout_seconds": 3
            }
            fs_task_adhoc = {"command": [sys.executable, "-c", "print('Ad-hoc FS task')"]}
            adhoc_fs_result = executor.run_in_sandbox(task_type="filesystem", task_details=fs_task_adhoc, sandbox_config=adhoc_fs_config)
            print(f"Filesystem task result (ad-hoc): {adhoc_fs_result}")
            # assert adhoc_fs_result["status"] == "success"
        else:
            logger.warning("Skipping adhoc filesystem task illustration.")

        # 6. Example of an unsupported task type
        logger.info("\n--- Running unsupported task type ---")
        unsupported_task_result = executor.run_in_sandbox(task_type="nonexistent_sandbox", task_details={})
        print(f"Unsupported task result: {unsupported_task_result}")
        assert unsupported_task_result["status"] == "error"
        assert "Unsupported sandbox task_type" in unsupported_task_result.get("error_message", "")


    logger.info("SandboxExecutor illustrative __main__ block completed.")
