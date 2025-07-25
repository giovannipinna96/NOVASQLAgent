import argparse
from smolagents import tool
from smolagents import CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# === TOOLS ===ù
@tool
def add_numbers(a: float, b: float) -> float:
    """
    Sum of two numbers. Returns the sum of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers. Returns the product of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a * b

@tool
def devide_numbers(a: float, b: float) -> float:
    """
    Devide two numbers. Returns the division of two numbers.

    Args:
      a: The first number.
      b: The second number.
    """
    return a / b

@tool
def giovannetor(a: float, b: float) -> float:
    """
    The 'giovannetor' function give is calculated as (a ** b) + 1.

    Args:
      a: The base number.
      b: The exponent.
    """
    return a ** b + 1


# tools = [
#     Tool(name="add_numbers", description="Add two numbers", func=add_numbers),
#     Tool(name="multiply_numbers", description="Multiply two numbers", func=multiply_numbers),
#     Tool(name="giovannetor", description="Raise a to the power of b", func=giovannetor),
# ]


# === BUILD LLM ===
def build_llm(model_id: str, local_model_path: str = None):
    if local_model_path:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    return pipe


# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--local_model_path", type=str, default=None)
    args = parser.parse_args()

    # Load LLM pipeline
    # llm_pipeline = build_llm(args.model, args.local_model_path)
    model = TransformersModel(model_id=args.local_model_path)
    agent = CodeAgent(tools=[add_numbers, multiply_numbers, giovannetor, devide_numbers], model=model)

    # Run the agent
    result = agent.run(args.query)

    print(f"\n========\nFinal Answer: {result}\n========")


if __name__ == "__main__":
    main()
