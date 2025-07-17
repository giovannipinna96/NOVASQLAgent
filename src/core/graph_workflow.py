# src/core/graph_workflow.py

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from ..llm.llm_factory import create_llm
from ..tools.langchain_tools import filesystem_tool, sql_tool
from ..prompting.prompts import agent_chat_prompt
from langchain_core.runnables import Runnable
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

# 2. Define the nodes for our graph
def agent_node(state: AgentState, agent: Runnable, name: str):
    """A node that runs the agent."""
    result = agent.invoke(state)
    return {"messages": [result]}

def tool_node(state: AgentState, agent: Runnable, name: str):
    """A node that runs the tools."""
    # This is a simplified tool node. A real implementation would need to
    # parse the tool calls from the last message and execute them.
    # For this example, we'll just return a message.
    return {"messages": [HumanMessage(content="[Tool output goes here]")]}

# 3. Define the main workflow class
class GraphWorkflow:
    def __init__(self, llm_provider='openai', llm_model='gpt-4'):
        self.llm = create_llm(llm_provider, llm_model)
        self.tools = [filesystem_tool, sql_tool]
        self.agent = create_tool_calling_agent(self.llm, self.tools, agent_chat_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools)
        self.graph = self._build_graph()
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.app = self.graph.compile(checkpointer=self.memory)

    def _build_graph(self):
        """Builds the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("agent", lambda state: agent_node(state, self.agent_executor, "agent"))
        graph.add_node("tools", lambda state: tool_node(state, self.agent_executor, "tools"))

        graph.set_entry_point("agent")

        # This is a simplified conditional edge. A real implementation would
        # check if the last message contains tool calls.
        graph.add_conditional_edges(
            "agent",
            lambda state: "tools" if "tool_calls" in state["messages"][-1].additional_kwargs else END,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", "agent")

        return graph

    def run(self, query: str):
        """Runs the agent with a given query."""
        config = {"configurable": {"thread_id": "1"}}
        # The input to the graph is a list of messages.
        # The initial message is the user's query.
        initial_messages = [HumanMessage(content=query)]

        # The graph will continue to run until it reaches an END state.
        # We can stream the output to see the intermediate steps.
        for event in self.app.stream({"messages": initial_messages}, config):
            for value in event.values():
                print("---")
                print(value)


if __name__ == '__main__':
    # Make sure to set your OPENAI_API_KEY environment variable for this example
    workflow = GraphWorkflow()

    # Example of an end-to-end simulated interaction
    print("--- Starting Agent Workflow ---")
    workflow.run("What is the current working directory?")
    print("\n--- Agent Workflow Finished ---")

    # To see the trace in LangSmith, make sure you have set up your
    # LangSmith environment variables as described in src/config/tracing.py
    print("\nTo view the trace, visit your LangSmith project dashboard.")
    print("Example trace URL: https://smith.langchain.com/o/YOUR_ORG/projects/YOUR_PROJECT?trace=YOUR_TRACE_ID")
