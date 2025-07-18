import argparse
from typing import Any, Dict

from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool, AgentType

# from mcp_client import MCPClient  # Assume wrapper/client for MCP server exists

# === Base Agent ===
class BaseLangGraphAgent:
    def __init__(self, model_id: str):
        self.llm = HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation")
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.prompt = PromptTemplate(input_variables=["input"], template="Answer the following question: {input}")

        self.agent = initialize_agent(
            tools=[],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def run(self, input_text: str) -> str:
        return self.agent.run(input_text)


# === Reasoning Agent (inherits from base) ===
class ReasoningAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Custom reasoning prompt or logic can go here


# === Judge Agent (inherits from base) ===
class JudgeAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.prompt = PromptTemplate(
            input_variables=["input"],
            template="Act as a judge. Evaluate the following statement and explain your reasoning: {input}"
        )

    def run(self, input_text: str) -> str:
        return self.agent.run(self.prompt.format(input=input_text))


# === React Agent with MCP Tool ===
class ReactAgent(BaseLangGraphAgent):
    def __init__(self, model_id: str, mcp_url: str):
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
    args = parser.parse_args()

    if args.agent == "base":
        agent = BaseLangGraphAgent(args.model)
    elif args.agent == "reasoning":
        agent = ReasoningAgent(args.model)
    elif args.agent == "judge":
        agent = JudgeAgent(args.model)
    elif args.agent == "react":
        if not args.mcp_url:
            raise ValueError("--mcp_url is required for the react agent")
        agent = ReactAgent(args.model, args.mcp_url)
    else:
        raise ValueError("Invalid agent type")

    # output = agent.run(args.query)
    # print("Agent output:\n", output)
    print("End of script.")
