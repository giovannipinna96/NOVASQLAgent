import os
from langsmith import Client

def setup_langsmith_tracing():
    """
    Configures LangSmith tracing.

    This function sets up the necessary environment variables for LangSmith.
    It's recommended to call this at the beginning of your application's entry point.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # It's recommended to set these environment variables in your shell or a .env file
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
    # os.environ["LANGCHAIN_PROJECT"] = "YOUR_PROJECT_NAME"

    # Example of how to create a client to interact with LangSmith programmatically
    # client = Client()
    # print("LangSmith tracing is set up.")

if __name__ == '__main__':
    # Example of how to use this function
    setup_langsmith_tracing()
    print("LangSmith environment variables are set. Make sure to set your API key and project name.")
    print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
    print(f"LANGCHAIN_ENDPOINT: {os.getenv('LANGCHAIN_ENDPOINT')}")
    print(f"LANGCHAIN_API_KEY: {'*' * 8 if os.getenv('LANGCHAIN_API_KEY') else 'Not set'}")
    print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
