# src/prompting/prompts.py

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# A simple prompt template for a generic task
generic_prompt = PromptTemplate.from_template(
    "Please provide a response to the following query: {query}"
)

# A more complex chat prompt template for an agent
agent_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

if __name__ == '__main__':
    # Example for generic_prompt
    formatted_generic_prompt = generic_prompt.format(query="What is the capital of France?")
    print("Formatted Generic Prompt:")
    print(formatted_generic_prompt)

    # Example for agent_chat_prompt
    from langchain_core.messages import AIMessage, HumanMessage

    chat_history = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I help you today?"),
    ]
    formatted_agent_prompt = agent_chat_prompt.format_messages(
        chat_history=chat_history,
        input="Can you tell me a joke?",
        agent_scratchpad=[]
    )
    print("\nFormatted Agent Chat Prompt:")
    print(formatted_agent_prompt)
