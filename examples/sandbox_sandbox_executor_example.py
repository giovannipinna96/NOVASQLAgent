# examples/sandbox_sandbox_executor_example.py

import sys
import os

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module and its potential dependencies (other sandboxes)
try:
    from src.sandbox.sandbox_executor import SandboxExecutor
    # We'll use the dummy versions of FileSystemSandbox and SQLSandbox for this example
    # as the actual ones might have complex setup.
    print("Attempting to use dummy sandboxes for SandboxExecutor example.")

    # --- DUMMY FileSystemSandbox (copied from its example for self-containment) ---
    class FileSystemSandbox:
        def __init__(self, base_path, **kwargs):
            self.base_path = base_path
            self._virtual_fs = {"/": {'type': 'directory', 'content': {}}}
            print(f"Dummy FileSystemSandbox for Executor initialized (base: {base_path}).")
        def _normalize_path(self, path):
            if not path.startswith("/"): path = "/" + path
            return os.path.normpath(path)
        def execute_fs_command(self, command, path, content=None, **kwargs):
            norm_path = self._normalize_path(path)
            print(f"FS Command: {command} on {norm_path}, Content: {content is not None}, Args: {kwargs}")
            if command == "create_file":
                parent_path = os.path.dirname(norm_path)
                name = os.path.basename(norm_path)
                # Simplified: assume parent exists and is a dir
                # In real code: find parent_path in self._virtual_fs and add name to its 'content'
                return {"status": "success", "message": f"File {norm_path} created (simulated).", "path": norm_path}
            elif command == "read_file":
                return {"status": "success", "content": f"Content of {norm_path} (simulated).", "path": norm_path}
            elif command == "list_directory":
                return {"status": "success", "files": ["file1.txt", "subdir/"], "path": norm_path}
            return {"status": "error", "message": "Unknown FS command or dummy limitation."}

    # --- DUMMY SQLSandbox (copied from its example for self-containment) ---
    class SQLSandbox:
        def __init__(self, db_config, **kwargs):
            self.db_config = db_config
            print(f"Dummy SQLSandbox for Executor initialized (config: {db_config}).")
        def execute_sql(self, sql_query, read_only=True):
            print(f"SQL Command: '{sql_query}', Read-only: {read_only}")
            if "SELECT" in sql_query.upper():
                return {"status": "success", "result_set": [{"col1": "data1", "col2": 100}], "columns": ["col1", "col2"]}
            elif not read_only and ("INSERT" in sql_query.upper() or "UPDATE" in sql_query.upper()):
                return {"status": "success", "rows_affected": 1}
            elif read_only and ("INSERT" in sql_query.upper() or "DROP" in sql_query.upper()):
                return {"status": "error", "message": "Write operation attempted in read-only mode."}
            return {"status": "error", "message": "Unsupported SQL or dummy limitation."}

except ImportError:
    print("Warning: Could not import SandboxExecutor from src.sandbox.sandbox_executor.")
    print("Using a dummy SandboxExecutor class and dummy sub-sandboxes for demonstration.")

    # --- DUMMY FileSystemSandbox (if not defined above due to import success) ---
    if 'FileSystemSandbox' not in globals():
        class FileSystemSandbox:
            def __init__(self, base_path, **kwargs):
                self.base_path = base_path
                self._virtual_fs = {"/": {'type': 'directory', 'content': {}}} # In-memory FS
                print(f"Dummy FileSystemSandbox for Executor initialized (base: {base_path}).")
            def _normalize_path(self, path):
                if not path.startswith("/"): path = "/" + path
                return os.path.normpath(path)
            # This method is what SandboxExecutor might expect
            def execute_fs_command(self, command, path, content=None, **kwargs):
                norm_path = self._normalize_path(path)
                print(f"FS Command via Executor: {command} on {norm_path}, Content: {content is not None}, Args: {kwargs}")
                if command == "create_file":
                    # Simplified: Assume parent exists. Add to virtual_fs.
                    # For dummy, just return success.
                    return {"status": "success", "message": f"File {norm_path} created (simulated).", "path": norm_path}
                elif command == "read_file":
                    return {"status": "success", "content": f"Content of {norm_path} (simulated).", "path": norm_path}
                elif command == "list_directory":
                     # Simplified: return fixed list
                    return {"status": "success", "files": ["sim_file1.txt", "sim_subdir/"], "path": norm_path}
                return {"status": "error", "message": "Unknown FS command or dummy limitation."}

    # --- DUMMY SQLSandbox (if not defined above) ---
    if 'SQLSandbox' not in globals():
        class SQLSandbox:
            def __init__(self, db_config, **kwargs):
                self.db_config = db_config
                print(f"Dummy SQLSandbox for Executor initialized (config: {db_config}).")
            # This method is what SandboxExecutor might expect
            def execute_sql(self, sql_query, read_only=True): # Renamed for clarity if executor calls this
                print(f"SQL Command via Executor: '{sql_query}', Read-only: {read_only}")
                if "SELECT" in sql_query.upper():
                    return {"status": "success", "result_set": [{"id": 1, "name": "Simulated Product"}], "columns": ["id", "name"]}
                elif not read_only and ("INSERT" in sql_query.upper() or "UPDATE" in sql_query.upper()):
                    return {"status": "success", "rows_affected": 1}
                elif read_only and ("INSERT" in sql_query.upper() or "DROP" in sql_query.upper()):
                    return {"status": "error", "message": "Write operation attempted in read-only mode (dummy SQL sandbox)."}
                return {"status": "error", "message": "Unsupported SQL or dummy limitation in SQL sandbox."}

    class SandboxExecutor:
        """
        Dummy SandboxExecutor class.
        """
        def __init__(self, fs_sandbox_instance=None, sql_sandbox_instance=None, config=None):
            self.fs_sandbox = fs_sandbox_instance if fs_sandbox_instance else FileSystemSandbox(base_path="/tmp/dummy_exec_fs")
            self.sql_sandbox = sql_sandbox_instance if sql_sandbox_instance else SQLSandbox(db_config={"type": "sqlite", "path": ":memory:"})
            self.config = config if config else {}
            print(f"Dummy SandboxExecutor initialized. Config: {self.config}")

        def execute_command(self, command_spec):
            """
            Dummy method to execute a command.
            Command spec could be a dictionary defining the type, command, and args.
            Example:
            { "type": "filesystem", "command": "create_file", "path": "/my_file.txt", "content": "Hello" }
            { "type": "sql", "command": "SELECT * FROM users", "read_only": True }
            """
            command_type = command_spec.get("type")
            print(f"\nExecutor received command: Type='{command_type}', Details='{command_spec}'")

            if command_type == "filesystem":
                if not self.fs_sandbox:
                    return {"status": "error", "message": "FileSystemSandbox not configured."}
                # Assume fs_sandbox has a generic 'execute_fs_command' or specific methods
                try:
                    # More generic approach:
                    fs_cmd = command_spec.get("command")
                    fs_path = command_spec.get("path")
                    fs_content = command_spec.get("content") # Optional
                    fs_kwargs = command_spec.get("options", {})
                    return self.fs_sandbox.execute_fs_command(command=fs_cmd, path=fs_path, content=fs_content, **fs_kwargs)
                except AttributeError as e: # Fallback to specific methods if execute_fs_command doesn't exist
                     return {"status": "error", "message": f"Filesystem command execution error (dummy): {e}"}
                except Exception as e:
                    return {"status": "error", "message": f"Generic error in FS command (dummy): {e}"}


            elif command_type == "sql":
                if not self.sql_sandbox:
                    return {"status": "error", "message": "SQLSandbox not configured."}
                sql_query_text = command_spec.get("query") # Changed from "command" to "query" for SQL
                sql_read_only = command_spec.get("read_only", True)
                # Assume sql_sandbox has an 'execute_sql' method
                try:
                    return self.sql_sandbox.execute_sql(sql_query=sql_query_text, read_only=sql_read_only)
                except Exception as e:
                    return {"status": "error", "message": f"SQL command execution error (dummy): {e}"}

            elif command_type == "python_code": # Example for a code execution sandbox
                 code_to_run = command_spec.get("code")
                 # This would be very complex to implement safely. Dummy just acknowledges.
                 print(f"Simulating execution of Python code (very dangerous in reality without extreme sandboxing):\n{code_to_run}")
                 return {"status": "success", "output": "Simulated output from Python code.", "errors": ""}

            else:
                print(f"Error: Unknown command type '{command_type}'")
                return {"status": "error", "message": f"Unknown command type: {command_type}"}


def main():
    print("--- SandboxExecutor Module Example ---")

    # Setup dummy sandboxes that the executor will use (if not using the real ones)
    # These are already instantiated within the dummy SandboxExecutor if none are provided.
    # For clarity, one could instantiate them here and pass them in.
    # dummy_fs = FileSystemSandbox(base_path="/tmp/example_fs_for_executor")
    # dummy_sql = SQLSandbox(db_config={"type": "sqlite", "database": "example_db.sqlite"})

    try:
        # executor = SandboxExecutor(fs_sandbox_instance=dummy_fs, sql_sandbox_instance=dummy_sql)
        executor = SandboxExecutor() # Uses internally created dummy sandboxes
    except NameError: # Fallback for dummy
        executor = SandboxExecutor()


    # Example 1: Execute a FileSystem command (create file)
    print("\n[Example 1: Execute FileSystem Create File Command]")
    fs_create_command = {
        "type": "filesystem",
        "command": "create_file",
        "path": "/projects/report.txt",
        "content": "This is a report generated via SandboxExecutor.",
        "options": {"overwrite": True}
    }
    result1 = executor.execute_command(fs_create_command)
    print(f"Executor Result: {result1}")

    # Example 2: Execute a FileSystem command (read file)
    print("\n[Example 2: Execute FileSystem Read File Command]")
    fs_read_command = {
        "type": "filesystem",
        "command": "read_file",
        "path": "/projects/report.txt"
    }
    result2 = executor.execute_command(fs_read_command)
    print(f"Executor Result: {result2}")
    # if result2.get('status') == 'success':
    #     print(f"Content read: {result2.get('content')}")

    # Example 3: Execute an SQL query (read-only)
    print("\n[Example 3: Execute SQL Select Query]")
    sql_select_command = {
        "type": "sql",
        "query": "SELECT user_id, username FROM users WHERE status = 'active'",
        "read_only": True
    }
    result3 = executor.execute_command(sql_select_command)
    print(f"Executor Result: {result3}")
    # if result3.get('status') == 'success':
    #     print(f"Query Results: {result3.get('result_set')}")

    # Example 4: Execute an SQL query (write operation, if allowed by dummy)
    print("\n[Example 4: Execute SQL Insert Query]")
    sql_insert_command = {
        "type": "sql",
        "query": "INSERT INTO logs (message) VALUES ('Executor test entry')",
        "read_only": False # Allow write
    }
    result4 = executor.execute_command(sql_insert_command)
    print(f"Executor Result: {result4}")

    # Example 5: Execute a potentially restricted SQL query in read-only mode
    print("\n[Example 5: Execute SQL Write Query in Read-Only Mode (should fail)]")
    sql_restricted_command = {
        "type": "sql",
        "query": "DROP TABLE users;",
        "read_only": True # Executor or SQLSandbox should prevent this
    }
    result5 = executor.execute_command(sql_restricted_command)
    print(f"Executor Result: {result5}")

    # Example 6: Execute a Python code snippet (if supported by the executor)
    if hasattr(executor, "execute_command"): # Check if dummy supports this type
        python_command_spec = {
            "type": "python_code",
            "code": "def greet(name):\n  return f'Hello, {name}!'\nprint(greet('World'))"
        }
        # Check if the dummy implementation can handle this type.
        # The current dummy SandboxExecutor has a basic handler for "python_code".
        print("\n[Example 6: Execute Python Code Snippet (Simulated)]")
        result6 = executor.execute_command(python_command_spec)
        print(f"Executor Result for Python code: {result6}")


    # Example 7: Unknown command type
    print("\n[Example 7: Execute Unknown Command Type]")
    unknown_command = {"type": "unknown_sandbox_type", "action": "do_something"}
    result7 = executor.execute_command(unknown_command)
    print(f"Executor Result: {result7}")

    print("\n--- SandboxExecutor Module Example Complete ---")

if __name__ == "__main__":
    main()
