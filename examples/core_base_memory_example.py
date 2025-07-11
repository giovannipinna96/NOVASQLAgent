# examples/model_memory_example.py

import sys
import os
import datetime # For timestamping example messages

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.model.memory import ConversationMemory # Assuming a class name
    # from src.model.memory import ShortTermMemory, LongTermMemoryConfig # Other potential classes
except ImportError:
    print("Warning: Could not import ConversationMemory from src.model.memory.")
    print("Using a dummy ConversationMemory class for demonstration purposes.")

    class ConversationMemory:
        """
        Dummy ConversationMemory class for demonstration if the actual module/class
        cannot be imported or its structure is unknown.
        """
        def __init__(self, memory_id="default_session", max_history_length=10, config=None):
            self.memory_id = memory_id
            self.max_history_length = max_history_length
            self.history = []
            self.config = config if config else {}
            print(f"Dummy ConversationMemory initialized for ID: {self.memory_id}, Max History: {self.max_history_length}, Config: {self.config}")

        def add_message(self, role, content, timestamp=None, metadata=None):
            """
            Dummy method to add a message to the conversation history.
            """
            if timestamp is None:
                timestamp = datetime.datetime.now().isoformat()

            message = {"role": role, "content": content, "timestamp": timestamp}
            if metadata:
                message["metadata"] = metadata

            self.history.append(message)
            print(f"\nMessage added: Role='{role}', Content='{content[:30]}...'")

            # Trim history if it exceeds max length
            if len(self.history) > self.max_history_length:
                self.history = self.history[-self.max_history_length:]
                print(f"History trimmed to last {self.max_history_length} messages.")

        def get_history(self, format_type="list_of_dicts", last_n=None):
            """
            Dummy method to retrieve the conversation history.
            """
            print(f"\nRetrieving history (format: {format_type}, last_n: {last_n})...")

            history_to_return = self.history
            if last_n is not None and last_n > 0:
                history_to_return = self.history[-last_n:]

            if format_type == "string":
                formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_to_return])
                print(f"Formatted History (string):\n{formatted_history}")
                return formatted_history
            elif format_type == "list_of_tuples":
                 formatted_history = [(msg['role'], msg['content']) for msg in history_to_return]
                 print(f"Formatted History (list of tuples): {formatted_history}")
                 return formatted_history

            # Default: list_of_dicts
            print(f"History (list of dicts): {history_to_return}")
            return history_to_return

        def clear_memory(self):
            """
            Dummy method to clear the conversation history.
            """
            self.history = []
            print("\nMemory cleared.")

        def get_last_user_message(self):
            """
            Dummy method to get the last message from the user.
            """
            for message in reversed(self.history):
                if message.get("role") == "user":
                    print(f"\nLast user message: {message}")
                    return message
            print("\nNo user message found in history.")
            return None

        def get_context_summary(self, max_tokens=100):
            """
            Dummy method to generate a summary of the conversation.
            """
            print(f"\nGenerating context summary (max_tokens: {max_tokens})...")
            if not self.history:
                return "No conversation history available."

            summary = "Conversation summary: "
            for msg in self.history:
                summary += f"{msg['role']}: {msg['content'][:20]}... " # Simplified summary

            summary = summary[:max_tokens]
            print(f"Context Summary (dummy): {summary}")
            return summary


def main():
    print("--- ConversationMemory Module Example ---")

    # Instantiate ConversationMemory
    # It might take a session ID, configuration for persistence, or max history size.
    try:
        memory_config = {"persistence_strategy": "in_memory", "max_tokens_per_entry": 500}
        chat_memory = ConversationMemory(memory_id="user123_chat_session", max_history_length=5, config=memory_config)
    except NameError: # Fallback for dummy
        chat_memory = ConversationMemory(memory_id="dummy_session", max_history_length=5)


    # Example 1: Add messages to memory
    print("\n[Example 1: Adding Messages]")
    chat_memory.add_message(role="user", content="Hello, I need help with my account.")
    chat_memory.add_message(role="assistant", content="Sure, I can help with that. What is your account ID?", metadata={"intent": "request_account_id"})
    chat_memory.add_message(role="user", content="My account ID is ACC10023.")
    chat_memory.add_message(role="assistant", content="Thank you. How can I assist you with account ACC10023 today?")

    # Example 2: Get conversation history
    print("\n[Example 2: Retrieving Full History]")
    full_history = chat_memory.get_history()
    # print("Full Conversation History:")
    # for msg in full_history:
    #     print(f"  {msg['timestamp']} - {msg['role']}: {msg['content']}")

    # Example 3: Get history in a different format (e.g., string for LLM prompt)
    if hasattr(chat_memory, "get_history"): # Already used, but good practice for optional features
        print("\n[Example 3: Retrieving History as String (last 2 messages)]")
        string_history = chat_memory.get_history(format_type="string", last_n=2)
        # print("String Formatted History (last 2):\n" + string_history)

        print("\n[Example 3b: Retrieving History as List of Tuples]")
        tuple_history = chat_memory.get_history(format_type="list_of_tuples")
        # print(f"Tuple Formatted History: {tuple_history}")


    # Example 4: Add more messages to see history trimming (if max_history_length is small)
    print("\n[Example 4: Adding More Messages to Trigger Trimming]")
    chat_memory.add_message(role="user", content="I want to know my current balance.")
    chat_memory.add_message(role="assistant", content="Okay, checking the balance for ACC10023...") # This is the 5th message
    chat_memory.add_message(role="user", content="Also, can I update my address?") # This is the 6th, should trigger trim if max_history_length=5

    current_history_after_trim = chat_memory.get_history()
    print(f"Number of messages after potential trim: {len(current_history_after_trim)}")
    # print("History after potential trim:")
    # for msg in current_history_after_trim:
    #     print(f"  {msg['role']}: {msg['content']}")


    # Example 5: Get last user message
    if hasattr(chat_memory, "get_last_user_message"):
        print("\n[Example 5: Get Last User Message]")
        last_user_msg = chat_memory.get_last_user_message()
        # if last_user_msg:
        #     print(f"Last User Message Content: {last_user_msg['content']}")

    # Example 6: Get context summary
    if hasattr(chat_memory, "get_context_summary"):
        print("\n[Example 6: Get Context Summary]")
        summary = chat_memory.get_context_summary(max_tokens=150)
        # print(f"Conversation Summary: {summary}")

    # Example 7: Clear memory
    print("\n[Example 7: Clearing Memory]")
    chat_memory.clear_memory()
    history_after_clear = chat_memory.get_history()
    print(f"Number of messages after clear: {len(history_after_clear)}")


    print("\n--- ConversationMemory Module Example Complete ---")

if __name__ == "__main__":
    main()
