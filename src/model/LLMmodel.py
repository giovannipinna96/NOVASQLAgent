import argparse
from typing import Any, Dict

from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import initialize_agent, Tool, AgentType

# from mcp_client import MCPClient  # Assume wrapper/client for MCP server exists

# === Base Agent ===

class BaseLangGraphAgent:
    def __init__(self, model_id: str, local_model_path: str = None):
        if local_model_path:
            print(f"Loading model from local path: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
            print(f"Model loaded: {model_id} from {local_model_path}")
            hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        else:
            self.llm = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation")
        self.prompt = PromptTemplate(input_variables=["input"], template="Answer the following question: {input}")

    def run(self, input_text: str) -> str:
        # Usa direttamente la pipeline LLM generica
        return self.llm(self.prompt.format(input=input_text))


# === Reasoning Agent (inherits from base) ===
class ReasoningAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str, local_model_path: str = None):
        super().__init__(model_id, local_model_path)
        # Custom reasoning prompt or logic can go here


# === Judge Agent (inherits from base) ===
class JudgeAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str, local_model_path: str = None):
        super().__init__(model_id, local_model_path)
        self.prompt = PromptTemplate(
            input_variables=["input"],
            template="Act as a judge. Evaluate the following statement and explain your reasoning: {input}"
        )

    def run(self, input_text: str) -> str:
        return self.agent.run(self.prompt.format(input=input_text))


# === React Agent with MCP Tool ===

class ReactAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str, mcp_url: str, local_model_path: str = None):
        if local_model_path:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
            hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        else:
            self.llm = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation")
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.prompt = PromptTemplate(input_variables=["input"], template="Answer the following question: {input}")

        # Set up MCP client (code sandbox MCP)
        # self.mcp = MCPClient(mcp_url)

        self.tools = [
            Tool(
                name="RunPythonInSandbox",
                func=self.run_code_in_sandbox,
                description="Executes Python code in a secure sandbox."
            )
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def run_code_in_sandbox(self, code: str) -> str:
        return self.mcp.run_code(code)

    def run(self, input_text: str) -> str:
        return self.agent.run(input_text)


# === Main testing logic ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["base", "reasoning", "judge", "react"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mcp_url", type=str, default="")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--local_model_path", type=str, default=None)
    args = parser.parse_args()
    
    print("Starting the agent with the following parameters:")
    print(f"Agent Type: {args.agent}")
    print(f"Model: {args.model}")
    print(f"Local model path: {args.local_model_path}")

    if args.agent == "base":
        agent = BaseLangGraphAgent(args.model, args.local_model_path)
    elif args.agent == "reasoning":
        agent = ReasoningAgent(args.model, args.local_model_path)
    elif args.agent == "judge":
        agent = JudgeAgent(args.model, args.local_model_path)
    elif args.agent == "react":
        if not args.mcp_url:
            raise ValueError("--mcp_url is required for the react agent")
        agent = ReactAgent(args.model, args.mcp_url, args.local_model_path)
    else:
        raise ValueError("Invalid agent type")

    output = agent.run(args.query)
    print("Agent output:\n", output)
    print("End of script.")
