"""
This example demonstrates how to use the SandboxExecutor to run tasks in different sandboxes.
"""

import logging
from src.tools.sandbox_executor import SandboxExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate SandboxExecutor usage.
    """
    executor = SandboxExecutor()

    # Run a filesystem task
    logger.info("--- Running a filesystem task ---")
    fs_task = {"command": ["ls", "-l"]}
    fs_result = executor.run_in_sandbox(task_type="filesystem", task_details=fs_task, config_name="default_fs")
    logger.info(f"Filesystem task result: {fs_result}")

    # Run a SQL task
    logger.info("--- Running a SQL task ---")
    sql_task = {
        "query": "CREATE TABLE users (id INTEGER, name TEXT); INSERT INTO users VALUES (1, 'Alice'); SELECT * FROM users;",
        "script": True,
    }
    sql_result = executor.run_in_sandbox(task_type="sql", task_details=sql_task, config_name="default_sql")
    logger.info(f"SQL task result: {sql_result}")

if __name__ == "__main__":
    main()
