# examples/langchain_agent_example.py

import os
from src.core.graph_workflow import GraphWorkflow
from src.config.tracing import setup_langsmith_tracing

def main():
    """
    An end-to-end example of the new LangGraph-based agent workflow.
    """
    # 1. Set up LangSmith tracing
    # Make sure you have set your LANGCHAIN_API_KEY and LANGCHAIN_PROJECT
    # environment variables.
    setup_langsmith_tracing()
    print("LangSmith tracing is configured.")

    # 2. Initialize the workflow
    # This example uses OpenAI's gpt-4, so ensure your OPENAI_API_KEY is set.
    print("\nInitializing agent workflow with OpenAI GPT-4...")
    try:
        workflow = GraphWorkflow(llm_provider='openai', llm_model='gpt-4')
        print("Workflow initialized successfully.")
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        print("Please ensure your OpenAI API key is set correctly.")
        return

    # 3. Run a sample interaction
    # The agent will use the filesystem tool to answer the question.
    query = "What are the files in the current working directory?"
    print(f"\n--- Running query: '{query}' ---")

    # The `run` method now streams the output of the graph execution.
    # You can see the agent's thoughts and tool calls in real-time.
    workflow.run(query)

    print("\n--- Query execution finished ---")

    # You can now check the LangSmith dashboard to see the full trace
    # of this interaction.
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"\nTrace is available in your LangSmith project: {project_name}")
    print("Navigate to https://smith.langchain.com/ to view it.")


if __name__ == "__main__":
    main()
