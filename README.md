# LangChain, LangGraph, and LangSmith Integration

This project has been refactored to use LangChain, LangGraph, and LangSmith for building, running, and observing LLM-powered agents.

## Architecture Overview

The new architecture is designed to be modular and extensible, leveraging the strengths of each framework:

- **LangChain:** Provides the core components for building with LLMs, including model wrappers, prompt templates, and tool abstractions.
- **LangGraph:** Used to define the agent's control flow as a state machine. This allows for complex, cyclical, and stateful interactions.
- **LangSmith:** Integrated for end-to-end tracing, debugging, and evaluation of the agent's performance.

### Key Modules

- `src/llm/llm_factory.py`: A factory for creating LLM instances from various providers (OpenAI, Anthropic, HuggingFace).
- `src/prompting/prompts.py`: Contains LangChain prompt templates used by the agent.
- `src/tools/langchain_tools.py`: Wraps sandboxed environments (filesystem, SQL) as LangChain `Tool` objects.
- `src/core/graph_workflow.py`: The heart of the agent, where the LangGraph state machine is defined and compiled.
- `src/evaluation/langsmith_evaluator.py`: A module for running evaluations on LangSmith to measure agent performance.

## Getting Started

### 1. Installation

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

To run the examples, you'll need to set up your environment variables. At a minimum, you'll need an OpenAI API key. For tracing with LangSmith, you'll also need your LangSmith API key and project details.

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="your-langsmith-project-name"
```

### 3. Running the Agent Example

The `examples/langchain_agent_example.py` script shows an end-to-end example of the new agent workflow.

```bash
python examples/langchain_agent_example.py
```

This will run a simple query against the agent and stream the output. You can then view the full trace in your LangSmith project.

### 4. Running Evaluations

The `src/evaluation/langsmith_evaluator.py` script demonstrates how to run an evaluation on LangSmith.

```bash
python src/evaluation/langsmith_evaluator.py
```

This will create a dataset on LangSmith, run an agent against it, and report the evaluation results.

## Development and Customization

### Adding New Tools

1. Create or modify a sandbox class in the `src/tools/` directory.
2. Wrap it as a `Tool` object in `src/tools/langchain_tools.py`.
3. Add the new tool to the `tools` list in `src/core/graph_workflow.py`.

### Changing the LLM

Modify the `llm_provider` and `llm_model` arguments when creating the `GraphWorkflow` instance in your script. The `llm_factory` supports 'openai', 'anthropic', and 'huggingface'.

### Customizing the Agent's Logic

The agent's behavior is defined in the `_build_graph` method of the `GraphWorkflow` class. You can add, remove, or modify the nodes and edges in the graph to change the agent's control flow.