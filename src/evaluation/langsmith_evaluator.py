# src/evaluation/langsmith_evaluator.py

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example
from typing import List, Dict, Any

# This is a placeholder for your agent's logic.
# In a real scenario, you would import your agent's entry point.
def your_agent_here(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    A placeholder for your agent's execution logic.
    This should take an input dictionary and return an output dictionary.
    """
    # For example, your agent might be the `run` method of your GraphWorkflow
    # from src.core.graph_workflow import GraphWorkflow
    # workflow = GraphWorkflow()
    # return {"output": workflow.run(inputs["input"])}

    # For this example, we'll just echo the input.
    return {"output": f"Agent response to: {inputs['input']}"}

def run_langsmith_evaluation(
    dataset_name: str,
    agent_to_evaluate,
    evaluators: List[Any],
    dataset_description: str = "A dataset for evaluating our agent.",
    project_name: str = "agent-evaluation"
):
    """
    Runs an evaluation on LangSmith.

    Args:
        dataset_name: The name of the dataset to use or create.
        agent_to_evaluate: The agent function to be evaluated.
        evaluators: A list of LangSmith evaluators to apply.
        dataset_description: A description for the dataset.
        project_name: The name of the LangSmith project for this evaluation.
    """
    client = Client()

    # 1. Create or load the dataset
    if not client.has_dataset(dataset_name=dataset_name):
        print(f"Creating dataset: {dataset_name}")
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=dataset_description,
        )
        # Add examples to the dataset
        client.create_examples(
            inputs=[
                {"input": "What is the capital of France?"},
                {"input": "What is 2+2?"},
            ],
            outputs=[
                {"output": "Paris"},
                {"output": "4"},
            ],
            dataset_id=dataset.id,
        )
    else:
        print(f"Using existing dataset: {dataset_name}")

    # 2. Run the evaluation
    print(f"Starting evaluation with project: {project_name}")
    evaluation_results = evaluate(
        agent_to_evaluate,
        dataset_name=dataset_name,
        evaluators=evaluators,
        experiment_prefix=project_name,
        metadata={"version": "1.0.0"},
    )
    print("Evaluation finished.")
    return evaluation_results

if __name__ == '__main__':
    # Example of how to run an evaluation
    from langsmith.evaluation import LangChainStringEvaluator

    # 1. Define the evaluators
    # This example uses a simple "correctness" evaluator that can be graded
    # in the LangSmith UI or via a custom grading function.
    string_evaluator = LangChainStringEvaluator(
        "correctness",
        # A simple lambda to grade the output.
        # In a real scenario, you might use an LLM as a judge.
        grading_function=lambda run, example: {
            "score": 1 if str(run.outputs["output"]) == str(example.outputs["output"]) else 0,
            "key": "correctness",
        }
    )

    # 2. Run the evaluation
    dataset_name = "my-agent-eval-dataset"
    results = run_langsmith_evaluation(
        dataset_name=dataset_name,
        agent_to_evaluate=your_agent_here,
        evaluators=[string_evaluator],
        project_name="example-agent-evaluation"
    )

    print("\nEvaluation Results:")
    print(results)
    print(f"\nView the results in your LangSmith project: {results.experiment_name}")
