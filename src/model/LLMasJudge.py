import argparse
from typing import Annotated, Sequence, TypedDict
import re

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.agents import Tool, create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, AnyMessage
from langchain.tools import tool
from langgraph.graph.message import add_messages
import torch

from langgraph.prebuilt import ToolNode
from langchain.prompts import PromptTemplate

react_prompt_template = """You are an intelligent agent that solves problems step by step using tools when needed.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}"""



# === Stato condiviso ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]

# === Funzione comune per costruire un LLM ===
def build_llm(model_id: str, local_model_path: str = None):
    if local_model_path:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,
                               device=0 if torch.cuda.is_available() else -1, return_full_text=False)
        return HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        return HuggingFacePipeline.from_model_id(model_id=model_id, task="text-generation",
                                                 device=0 if torch.cuda.is_available() else -1)

# === TOOLS ===
@tool
def add_numbers(a: float, b: float) -> float:
    """Returns the sum of two numbers."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Returns the product of two numbers."""
    return a * b

@tool
def giovannetor(a: float, b: float) -> float:
    """Returns the giovannetor number."""
    return a ** b

tools = [add_numbers, multiply_numbers, giovannetor]

# === AGENT FACTORIES ===
def base_agent_node(model_id: str, local_model_path: str = None) -> RunnableLambda:
    llm = build_llm(model_id, local_model_path)
    def _run(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are a helpful assistant. Answer the user's question.")
        user_msgs = state["messages"]
        result = llm.invoke([system_prompt] + user_msgs)
        return {"messages": user_msgs + [result]}
    return RunnableLambda(_run)

def reasoning_agent_node(model_id: str, local_model_path: str = None) -> RunnableLambda:
    llm = build_llm(model_id, local_model_path)
    def _run(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are a reasoning agent. Think step by step and explain your reasoning before answering.")
        user_msgs = state["messages"]
        result = llm.invoke([system_prompt] + user_msgs)
        return {"messages": user_msgs + [result]}
    return RunnableLambda(_run)

def judge_agent_node(model_id: str, local_model_path: str = None) -> RunnableLambda:
    llm = build_llm(model_id, local_model_path)
    def _run(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="Act as a judge. Evaluate the following statement and explain your reasoning.")
        user_msgs = state["messages"]
        result = llm.invoke([system_prompt] + user_msgs)
        return {"messages": user_msgs + [result]}
    return RunnableLambda(_run)

def mcp_agent_node(model_id: str, mcp_func, local_model_path: str = None) -> RunnableLambda:
    llm = build_llm(model_id, local_model_path)
    def _run(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are an agent that can use an MCP server. (MCP logic not implemented yet)")
        user_msgs = state["messages"]
        result = llm.invoke([system_prompt] + user_msgs)
        return {"messages": user_msgs + [result]}
    return RunnableLambda(_run)

# === React agent node ===
def react_agent_node(model_id: str, local_model_path: str = None):
    llm = build_llm(model_id, local_model_path)

    # Crea il prompt esplicito con tool descriptions
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    print()
    print(f"Tool descriptions: {tool_descriptions}")
    print(f"Tool names: {tool_names}")
    print()

    prompt = PromptTemplate.from_template(react_prompt_template).partial(
        tools=tool_descriptions,
        tool_names=tool_names
    )

    # Crea l'agente ReAct
    agent = create_react_agent(llm, tools, prompt)

    def _run(state: AgentState) -> AgentState:
        user_input = state["messages"][-1].content
        output = agent.invoke({
            "input": user_input,
            "intermediate_steps": []
            })
        if hasattr(output, "return_values") and "output" in output.return_values:
            final_text = output.return_values["output"]
        else:
            final_text = str(output)

        state["messages"].append(AIMessage(content=final_text))
        return state

    return RunnableLambda(_run)


# def react_tool_node(state: AgentState) -> AgentState:
#     last = state["messages"][-1].content
#     match = re.search(r"Action: (.*?)\nAction Input: (.*)", last, re.DOTALL)
#     if not match:
#         return state

#     action_name = match.group(1).strip()
#     action_input = match.group(2).strip().strip('"')
    
#     print(f"Action: {action_name}, Input: {action_input}")

#     for tool in tools:
#         if tool.name == action_name:
#             observation = tool.run(action_input)
#             break
#     else:
#         observation = f"Tool {action_name} not found."

#     new_content = f"Observation: {observation}\nThought:"
#     state["messages"].append(AIMessage(content=new_content))
#     return state

def react_tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1].content

    match = re.search(r"Action:\s*(\w+)\s*Action Input:\s*(.*)", last, re.DOTALL)
    if not match:
        return state

    action_name = match.group(1).strip()
    action_input_raw = match.group(2).strip()

    args = re.findall(r"[-+]?\d*\.\d+|\d+", action_input_raw)  
    args = [float(x) for x in args]

    print(f"🛠️ Calling tool: {action_name} with input {args}")

    for tool in tools:
        if tool.name == action_name:
            try:
                observation = tool.run(*args)
            except Exception as e:
                observation = f"Tool call error: {e}"
            break
    else:
        observation = f"Tool '{action_name}' not found."

    new_content = f"Observation: {observation}\nThought:"
    state["messages"].append(AIMessage(content=new_content))
    return state


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1].content
    return "continue" if "Action:" in last else "end"

def create_graph(agent_type: str, model_id: str, local_model_path: str = None, mcp_func=None):
    builder = StateGraph(AgentState)
    if agent_type == "base":
        builder.add_node("agent", base_agent_node(model_id, local_model_path))
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
    elif agent_type == "reasoning":
        builder.add_node("agent", reasoning_agent_node(model_id, local_model_path))
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
    elif agent_type == "judge":
        builder.add_node("agent", judge_agent_node(model_id, local_model_path))
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
    elif agent_type == "react":
        builder.add_node("agent", react_agent_node(model_id, local_model_path))
        builder.add_node("tools", ToolNode("tools", [add_numbers, multiply_numbers, giovannetor]))
        # builder.add_node("tools", react_tool_node)
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {
            "continue": "tools",
            "end": END,
        })
        builder.add_edge("tools", "agent")
    elif agent_type == "mcp":
        builder.add_node("agent", mcp_agent_node(model_id, mcp_func, local_model_path))
        builder.set_entry_point("agent")
        builder.add_edge("agent", END)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return builder.compile()

# === Dummy MCP function ===
def dummy_mcp_run(code: str) -> str:
    return f"[MCP SIMULATION]: Executed code: {code}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["base", "reasoning", "judge", "react", "mcp"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mcp_url", type=str, default="")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--local_model_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Running agent type: {args.agent}")
    graph = create_graph(
        agent_type=args.agent,
        model_id=args.model,
        local_model_path=args.local_model_path,
        mcp_func=dummy_mcp_run if args.agent == "mcp" else None
    )

    user_msg = HumanMessage(content=args.query)
    result = graph.invoke({"messages": [user_msg]})
    print("\nAgent output:")
    if "messages" in result:
        for m in result["messages"]:
            try:
                m.pretty_print()
            except Exception:
                print(m)
    else:
        print(result)

