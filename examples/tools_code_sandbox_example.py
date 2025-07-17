"""
This example demonstrates how to use the CodeSandbox to execute commands in a Docker container.
"""

import logging
from src.tools.code_sandbox import CodeSandbox

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate CodeSandbox usage.
    """
    # Initialize the sandbox
    sandbox = CodeSandbox()
    container_id = sandbox.initialize()

    if not container_id:
        logger.error("Failed to initialize sandbox.")
        return

    # Execute a simple command
    logger.info("--- Running a simple command ---")
    result = sandbox.execute(["echo", "Hello from the sandbox!"])
    logger.info(f"Result: {result}")

    # Write a file and then read it
    logger.info("--- Writing and reading a file ---")
    file_content = "This is a test file."
    sandbox.write_file("test.txt", file_content)
    result = sandbox.execute(["cat", "test.txt"])
    logger.info(f"File content: {result}")

    # Stop the sandbox
    sandbox.stop()

if __name__ == "__main__":
    main()
