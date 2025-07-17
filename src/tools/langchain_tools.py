# src/tools/langchain_tools.py

from langchain.tools import Tool
from .filesystem_sandbox import FilesystemSandbox
from .sql_sandbox import SQLSandbox

# It's assumed that FilesystemSandbox and SQLSandbox are in the same directory
# and have methods like 'run' or similar that can be wrapped by the Tool object.

# For this example, let's assume FilesystemSandbox has a 'run_command' method
# and SQLSandbox has an 'execute_query' method.

# Filesystem Tool
fs_sandbox = FilesystemSandbox(working_dir="sandbox_workspace")
filesystem_tool = Tool(
    name="filesystem_sandbox",
    func=fs_sandbox.execute_command,
    description="Executes a command in a secure filesystem sandbox. Useful for file operations, code execution, and other shell commands.",
)

# SQL Tool
# The SQLSandbox might need to be initialized with a database connection
sql_sandbox = SQLSandbox()  # Add connection details if needed
sql_tool = Tool(
    name="sql_sandbox",
    func=sql_sandbox.execute_query,
    description="Executes a SQL query against the connected database. Useful for data retrieval and analysis.",
)

if __name__ == '__main__':
    # Example usage of the tools

    # 1. Filesystem Tool
    print("--- Filesystem Tool Example ---")
    try:
        # List files in the root directory
        result = filesystem_tool.run("ls -l /")
        print("Result of 'ls -l /':")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

    # 2. SQL Tool
    print("\n--- SQL Tool Example ---")
    try:
        # This will fail without a database connection, but it demonstrates the concept.
        # To run this, you would need to set up the SQLSandbox with a valid DB.
        # For example: sql_sandbox = SQLSandbox(db_path=':memory:')
        sql_query = "SELECT * FROM users LIMIT 5;"
        result = sql_tool.run(sql_query)
        print(f"Result of '{sql_query}':")
        print(result)
    except Exception as e:
        print(f"An error occurred with the SQL tool: {e}")
        print("This is expected if the database is not configured.")
