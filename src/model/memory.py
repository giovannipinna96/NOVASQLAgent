"""
This file is dedicated to implementing a robust memory management system for handling LLM-driven conversations.
The primary goal is to support agent-based architectures in which each agent maintains its own memory of messages
(sent and received) along with metadata and structural utilities to support advanced conversational workflows.

CORE REQUIREMENTS AND FUNCTIONALITY:

1. CONVERSATION MEMORY MANAGEMENT:
    - Implement a system (class or module) that stores conversations as structured sequences of messages.
    - Each message should contain:
        • The message content (string).
        • A unique message identifier (UUID or similar).
        • The direction of the message (incoming or outgoing).
        • The name of the LLM model that generated or received the message (fetched from the `LLMmodel.py` class attribute `name`).
    - Each agent should have its own instance of conversation memory to maintain individualized histories.

2. MESSAGE DELETION:
    - Provide methods to easily delete:
        • Specific messages by ID.
        • All messages before or after a certain timestamp or index.
        • Messages matching specific filters (e.g., LLM name, content keywords, etc.).

3. CONVERSATION SUMMARIZATION:
    - When a conversation exceeds a predefined maximum token length (default value settable via class constant or constructor argument),
      the system must:
        • Tokenize the conversation using the tokenizer associated with the LLM in use.
        • Automatically summarize the full conversation using a summarization-capable LLM.
        • Replace the full memory with a single summarized message to preserve token budget.
    - Token counting must be accurate and compatible with both local (HuggingFace) and API-based models.

4. JSON EXPORT AND IMPORT:
    - Implement methods to:
        • Export the entire conversation memory to a JSON file or string.
        • Reconstruct a conversation memory instance from a JSON file or JSON-formatted string.
    - Ensure exported format preserves all necessary fields for full rehydration (message ID, LLM name, timestamps, etc.).

5. TEMPORAL SNAPSHOT STORAGE:
    - Maintain a snapshot log of all conversation states across time.
    - Every modification (insertion, deletion, summarization) should create a versioned snapshot.
    - Provide a method to restore the conversation memory to any prior snapshot by timestamp or snapshot ID.

6. SYSTEM INTEGRATION:
    - The memory management system must be fully compatible and tightly integrated with:
        • `LLMmodel.py`: to access model names, tokenizers, and generation capabilities.
        • `prompt_template_manager.py`: to manage and optionally format prompts for summarization or reconstruction.

7. CODE QUALITY AND BEST PRACTICES:
    - Follow all standard Python best practices:
        • Full type annotations and PEP8-compliant formatting.
        • Use of Python `dataclasses` or `TypedDict` for clear message structuring.
        • Meaningful logging and exception handling.
        • Modularity and reusability for agent- or multi-agent systems.
    - Use docstrings for all public classes and methods.
    - Ensure extensibility to support future features like message encryption, memory pruning strategies, or hierarchical memory.

This module forms the backbone of long-term memory for agent-based LLM applications and is designed to support
complex interaction histories with persistence, summarization, and recovery capabilities.
"""
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

# Assuming LLMmodel.py and prompt_template_manager.py exist in the same directory or are importable
# For now, we'll use placeholder types/functions for LLM interaction.
# from .LLMmodel import BaseLLM  # Placeholder for LLM model integration
# from .prompt_template_manager import PromptTemplateManager # Placeholder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageDirection(Enum):
    """Indicates the direction of a message."""
    INCOMING = "incoming"  # Message received by the agent (e.g., from user or another LLM)
    OUTGOING = "outgoing"  # Message sent by the agent (e.g., LLM response)

@dataclass
class Message:
    """Represents a single message in a conversation."""
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    direction: MessageDirection = MessageDirection.OUTGOING
    llm_model_name: Optional[str] = None  # Name of the LLM that generated/received this
    metadata: Dict[str, Any] = field(default_factory=dict) # For additional info

    def to_dict(self) -> Dict[str, Any]:
        """Converts the message to a dictionary."""
        data = asdict(self)
        data['direction'] = self.direction.value # Ensure enum is serialized as string
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Creates a Message object from a dictionary."""
        data['direction'] = MessageDirection(data['direction']) # Deserialize string to enum
        return cls(**data)


# Placeholder types for LLM interaction - replace with actual imports when LLMmodel is ready
class PlaceholderLLM:
    """Placeholder for an LLM model, to be replaced by actual LLMmodel.BaseLLM."""
    def __init__(self, name: str, tokenizer: Optional[Callable[[str], List[int]]] = None, max_tokens: int = 4096):
        self.name = name
        self.tokenizer = tokenizer if tokenizer else lambda text: list(range(len(text.split()))) # Dummy tokenizer
        self.max_tokens = max_tokens

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Placeholder summarization function."""
        logger.info(f"PlaceholderLLM ({self.name}): Summarizing text (first 100 chars): {text[:100]}...")
        return f"Summary of: {text[:max_length//2]}..."

    def count_tokens(self, text: str) -> int:
        """Placeholder token counting function."""
        return len(self.tokenizer(text))

# Placeholder for PromptTemplateManager
class PlaceholderPromptManager:
    def format_prompt(self, data: Dict[str,str]) -> str:
        return f"Formatted prompt with: {data}"


@dataclass
class ConversationSnapshot:
    """Represents a snapshot of the conversation memory at a specific time."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    messages: List[Message] = field(default_factory=list)
    description: str = "Snapshot" # e.g., "After adding message X", "After summarization"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "messages": [msg.to_dict() for msg in self.messages],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSnapshot':
        return cls(
            snapshot_id=data['snapshot_id'],
            timestamp=data['timestamp'],
            messages=[Message.from_dict(msg_data) for msg_data in data['messages']],
            description=data.get('description', "Snapshot from dict")
        )


class ConversationMemory:
    """
    Manages conversation history for an agent, including storage, retrieval,
    deletion, summarization, and versioning.
    """

    def __init__(
        self,
        agent_id: str,
        llm_model: Optional[Any] = None, # Should be BaseLLM instance from LLMmodel.py
        # prompt_manager: Optional[Any] = None, # Should be PromptTemplateManager instance
        max_token_length: Optional[int] = None, # Default handled by llm_model or a constant
        summarization_llm: Optional[Any] = None, # Specific LLM for summarization
    ):
        """
        Initializes ConversationMemory.

        Args:
            agent_id: A unique identifier for the agent this memory belongs to.
            llm_model: The primary LLM model instance associated with this memory.
                       Used for tokenization and potentially summarization if no specific
                       summarization_llm is provided.
            # prompt_manager: An instance of PromptTemplateManager for formatting summarization prompts.
            max_token_length: The maximum number of tokens before summarization is triggered.
                              If None, uses llm_model.max_tokens or a default.
            summarization_llm: An optional, specific LLM model for summarization tasks.
                               If None, llm_model is used.
        """
        self.agent_id: str = agent_id
        self.messages: List[Message] = []
        self.snapshots: List[ConversationSnapshot] = []

        # Use placeholder if no actual LLM model is provided during early development
        self.llm_model = llm_model if llm_model else PlaceholderLLM(name="default_llm_for_memory")
        self.summarization_llm = summarization_llm if summarization_llm else self.llm_model
        # self.prompt_manager = prompt_manager if prompt_manager else PlaceholderPromptManager()

        # Determine max_token_length
        if max_token_length is not None:
            self.max_token_length: int = max_token_length
        elif hasattr(self.llm_model, 'max_tokens'): # Check if actual LLM has this attribute
             self.max_token_length: int = self.llm_model.max_tokens
        else:
            self.max_token_length: int = 4096 # Default fallback
        
        logger.info(f"ConversationMemory for agent '{agent_id}' initialized. Max tokens: {self.max_token_length}")
        self._create_snapshot("Initial state")

    def _create_snapshot(self, description: str) -> None:
        """Creates a snapshot of the current message list."""
        # Deep copy messages to avoid modification of snapshot by future changes to self.messages
        snapshot_messages = [Message.from_dict(msg.to_dict()) for msg in self.messages]
        snapshot = ConversationSnapshot(messages=snapshot_messages, description=description)
        self.snapshots.append(snapshot)
        logger.debug(f"Created snapshot {snapshot.snapshot_id}: {description}")

    def add_message(self, content: str, direction: MessageDirection, llm_model_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Adds a new message to the conversation.

        Args:
            content: The text content of the message.
            direction: The direction of the message (INCOMING or OUTGOING).
            llm_model_name: Name of the LLM model involved. Defaults to primary LLM if OUTGOING.
            metadata: Optional dictionary for additional message data.

        Returns:
            The created Message object.
        """
        if direction == MessageDirection.OUTGOING and llm_model_name is None:
            llm_model_name = self.llm_model.name

        message = Message(
            content=content,
            direction=direction,
            llm_model_name=llm_model_name,
            metadata=metadata or {}
        )
        self.messages.append(message)
        logger.info(f"Added message {message.message_id} ({direction.value}) to agent '{self.agent_id}'.")
        self._create_snapshot(f"Added message {message.message_id}")
        self._check_and_summarize()
        return message

    def get_messages(self) -> List[Message]:
        """Returns all messages in the conversation."""
        return self.messages

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Retrieves a message by its ID."""
        for msg in self.messages:
            if msg.message_id == message_id:
                return msg
        logger.warning(f"Message with ID '{message_id}' not found.")
        return None

    def delete_message_by_id(self, message_id: str) -> bool:
        """Deletes a message by its ID."""
        original_len = len(self.messages)
        self.messages = [msg for msg in self.messages if msg.message_id != message_id]
        if len(self.messages) < original_len:
            logger.info(f"Deleted message with ID '{message_id}'.")
            self._create_snapshot(f"Deleted message {message_id}")
            return True
        logger.warning(f"Failed to delete message: ID '{message_id}' not found.")
        return False

    def delete_messages_before_timestamp(self, timestamp_iso: str) -> int:
        """Deletes all messages before a given ISO timestamp."""
        try:
            cutoff_dt = datetime.fromisoformat(timestamp_iso)
        except ValueError:
            logger.error(f"Invalid ISO timestamp format: {timestamp_iso}")
            return 0

        deleted_count = 0
        new_messages = []
        for msg in self.messages:
            msg_dt = datetime.fromisoformat(msg.timestamp)
            if msg_dt >= cutoff_dt:
                new_messages.append(msg)
            else:
                deleted_count += 1
        
        if deleted_count > 0:
            self.messages = new_messages
            logger.info(f"Deleted {deleted_count} messages before {timestamp_iso}.")
            self._create_snapshot(f"Deleted {deleted_count} messages before {timestamp_iso}")
        return deleted_count

    def delete_messages_by_filter(self, filter_func: Callable[[Message], bool]) -> int:
        """
        Deletes messages based on a custom filter function.

        Args:
            filter_func: A function that takes a Message object and returns True if it should be deleted.

        Returns:
            The number of messages deleted.
        """
        original_len = len(self.messages)
        self.messages = [msg for msg in self.messages if not filter_func(msg)]
        deleted_count = original_len - len(self.messages)
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} messages based on filter.")
            self._create_snapshot(f"Deleted {deleted_count} messages by filter")
        return deleted_count
        
    def _get_conversation_text(self) -> str:
        """Concatenates all message contents into a single string."""
        return "\n".join([f"{msg.direction.value.capitalize()} ({msg.llm_model_name or 'N/A'}): {msg.content}" for msg in self.messages])

    def _count_current_tokens(self) -> int:
        """Counts the total tokens in the current conversation."""
        if not self.messages:
            return 0
        full_text = self._get_conversation_text()
        # Ensure llm_model has a count_tokens method
        if hasattr(self.llm_model, 'count_tokens') and callable(self.llm_model.count_tokens):
            return self.llm_model.count_tokens(full_text)
        else: # Fallback to a very basic split-based token count if method is missing
            logger.warning(f"LLM model {self.llm_model.name} missing 'count_tokens' method. Using basic word count.")
            return len(full_text.split())


    def _check_and_summarize(self) -> None:
        """Checks if token limit is exceeded and triggers summarization if needed."""
        current_tokens = self._count_current_tokens()
        if current_tokens > self.max_token_length:
            logger.info(f"Token limit exceeded ({current_tokens}/{self.max_token_length}). Summarizing conversation for agent '{self.agent_id}'.")
            self.summarize_conversation()


    def summarize_conversation(self) -> None:
        """
        Summarizes the entire conversation using the summarization LLM and replaces
        the current message list with a single summary message.
        """
        if not self.messages:
            logger.info("No messages to summarize.")
            return

        full_conversation_text = self._get_conversation_text()
        
        # Ensure summarization_llm has a summarize method
        if not hasattr(self.summarization_llm, 'summarize') or not callable(self.summarization_llm.summarize):
            logger.error(f"Summarization LLM ({self.summarization_llm.name}) does not have a 'summarize' method. Skipping summarization.")
            return

        try:
            # Here, you might use prompt_manager to format a specific prompt for summarization
            # summary_prompt = self.prompt_manager.format_prompt({"conversation_text": full_conversation_text})
            # summary_content = self.summarization_llm.run(summary_prompt) # Assuming LLM has a .run() method
            
            # Using the placeholder .summarize() method for now
            summary_content = self.summarization_llm.summarize(full_conversation_text)
            
            summary_message = Message(
                content=f"Summary of prior conversation: {summary_content}",
                direction=MessageDirection.INCOMING, # System-generated summary
                llm_model_name=self.summarization_llm.name,
                metadata={"is_summary": True, "original_message_count": len(self.messages)}
            )
            
            self.messages = [summary_message]
            logger.info(f"Conversation summarized. New message count: {len(self.messages)}")
            self._create_snapshot("Conversation summarized")
        except Exception as e:
            logger.error(f"Error during conversation summarization: {e}")
            # Optionally, retain original messages or handle error more gracefully
            # For now, we'll just log and not change messages if summarization fails.

    def to_json(self, filepath: Union[str, Path]) -> None:
        """Exports the conversation memory (messages and snapshots) to a JSON file."""
        data = {
            "agent_id": self.agent_id,
            "max_token_length": self.max_token_length,
            "messages": [msg.to_dict() for msg in self.messages],
            "snapshots": [snap.to_dict() for snap in self.snapshots],
            # We might need to store info about the LLM models if they are critical for rehydration
            # For now, assuming they are passed in again on load
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Conversation memory for agent '{self.agent_id}' exported to {filepath}")
        except IOError as e:
            logger.error(f"Error exporting conversation memory to JSON: {e}")
            raise

    @classmethod
    def from_json(
        cls,
        filepath: Union[str, Path],
        llm_model: Optional[Any] = None, # BaseLLM
        # prompt_manager: Optional[Any] = None, # PromptTemplateManager
        summarization_llm: Optional[Any] = None # BaseLLM
    ) -> 'ConversationMemory':
        """
        Reconstructs a ConversationMemory instance from a JSON file.
        LLM models and prompt manager need to be provided as they are not serialized.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found for import: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath}: {e}")
            raise
        except IOError as e:
            logger.error(f"Error reading JSON file {filepath}: {e}")
            raise

        memory = cls(
            agent_id=data['agent_id'],
            llm_model=llm_model,
            # prompt_manager=prompt_manager,
            max_token_length=data.get('max_token_length'), # Use .get for backward compatibility
            summarization_llm=summarization_llm
        )
        memory.messages = [Message.from_dict(msg_data) for msg_data in data['messages']]
        
        # Reconstruct snapshots carefully
        memory.snapshots = [] # Clear initial snapshot created by constructor
        for snap_data in data.get("snapshots", []): # Handle if snapshots key is missing
            try:
                snapshot = ConversationSnapshot.from_dict(snap_data)
                memory.snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Could not reconstruct snapshot from data: {snap_data}. Error: {e}")

        logger.info(f"Conversation memory for agent '{data['agent_id']}' imported from {filepath}. Loaded {len(memory.messages)} messages and {len(memory.snapshots)} snapshots.")
        return memory

    def restore_from_snapshot(self, snapshot_id: Optional[str] = None, timestamp_iso: Optional[str] = None) -> bool:
        """
        Restores the conversation memory to a specific snapshot.
        Can specify either snapshot_id or timestamp_iso. If timestamp_iso, uses the latest snapshot before or at that time.
        If both are None, restores to the latest snapshot.

        Args:
            snapshot_id: The ID of the snapshot to restore.
            timestamp_iso: The ISO timestamp to restore to (latest snapshot at or before this time).

        Returns:
            True if restoration was successful, False otherwise.
        """
        target_snapshot: Optional[ConversationSnapshot] = None

        if snapshot_id:
            target_snapshot = next((s for s in self.snapshots if s.snapshot_id == snapshot_id), None)
            if not target_snapshot:
                logger.error(f"Snapshot with ID '{snapshot_id}' not found.")
                return False
        elif timestamp_iso:
            try:
                cutoff_dt = datetime.fromisoformat(timestamp_iso)
                # Find the latest snapshot whose timestamp is less than or equal to cutoff_dt
                eligible_snapshots = [s for s in self.snapshots if datetime.fromisoformat(s.timestamp) <= cutoff_dt]
                if eligible_snapshots:
                    target_snapshot = max(eligible_snapshots, key=lambda s: datetime.fromisoformat(s.timestamp))
                else:
                    logger.warning(f"No snapshot found at or before timestamp {timestamp_iso}.")
                    return False
            except ValueError:
                logger.error(f"Invalid ISO timestamp format for restoration: {timestamp_iso}")
                return False
        elif self.snapshots: # Restore to latest if no specific id/time given
             target_snapshot = self.snapshots[-1]
        else:
            logger.warning("No snapshots available to restore from.")
            return False
            
        if target_snapshot:
            # Deep copy messages from snapshot
            self.messages = [Message.from_dict(msg.to_dict()) for msg in target_snapshot.messages]
            description = f"Restored from snapshot {target_snapshot.snapshot_id} ({target_snapshot.description} at {target_snapshot.timestamp})"
            logger.info(description)
            # Create a new snapshot indicating this restoration event
            self._create_snapshot(description)
            return True
        
        logger.error("Failed to identify a snapshot for restoration.")
        return False

    def get_snapshot_log(self) -> List[Dict[str, str]]:
        """Returns a log of all snapshots (ID, timestamp, description)."""
        return [{"id": s.snapshot_id, "timestamp": s.timestamp, "description": s.description} for s in self.snapshots]


if __name__ == '__main__':
    # Example Usage
    # Dummy LLM for testing token counting and summarization
    def dummy_tokenizer(text: str) -> List[int]:
        return list(range(len(text.split()))) # Each word is a token

    test_llm = PlaceholderLLM(name="test_summarizer_llm", tokenizer=dummy_tokenizer, max_tokens=50)
    
    # Initialize memory
    agent_mem = ConversationMemory(agent_id="test_agent_001", llm_model=test_llm, max_token_length=30) # Lower max_token_length for testing summarization

    # Add messages
    msg1 = agent_mem.add_message("Hello, this is the first user message.", MessageDirection.INCOMING, llm_model_name="User")
    msg2 = agent_mem.add_message("Hi User! This is my response, quite elaborate to make it long enough.", MessageDirection.OUTGOING)
    msg3 = agent_mem.add_message("Okay, another message from the user to make sure we have enough content.", MessageDirection.INCOMING, llm_model_name="User")
    
    print(f"\nInitial messages ({len(agent_mem.get_messages())}):")
    for msg in agent_mem.get_messages():
        print(f"  - {msg.content[:50]}... (Tokens: {test_llm.count_tokens(msg.content)})")

    # This should trigger summarization if max_token_length is low enough
    # Let's add one more to definitely exceed 30 "tokens" (words in this dummy setup)
    msg4 = agent_mem.add_message("This is the final message which should push it over the token limit for sure.", MessageDirection.INCOMING, llm_model_name="User")

    print(f"\nMessages after potential summarization ({len(agent_mem.get_messages())}):")
    for msg in agent_mem.get_messages():
        print(f"  - {msg.content[:100]}...")
    
    assert len(agent_mem.get_messages()) == 1 # Should be summarized into one message
    assert "Summary of prior conversation" in agent_mem.get_messages()[0].content

    # Test deletion
    agent_mem.add_message("A new message after summary.", MessageDirection.OUTGOING)
    msg_to_delete = agent_mem.add_message("This message will be deleted.", MessageDirection.INCOMING, llm_model_name="User")
    assert len(agent_mem.get_messages()) == 3
    agent_mem.delete_message_by_id(msg_to_delete.message_id)
    assert len(agent_mem.get_messages()) == 2
    assert not agent_mem.get_message_by_id(msg_to_delete.message_id)

    # Test JSON export/import
    json_file_path = Path("test_agent_memory.json")
    agent_mem.to_json(json_file_path)
    
    # Create new LLM instance for loading, as it's not serialized
    loaded_llm = PlaceholderLLM(name="loaded_llm", tokenizer=dummy_tokenizer)
    loaded_mem = ConversationMemory.from_json(json_file_path, llm_model=loaded_llm)
    
    assert loaded_mem.agent_id == agent_mem.agent_id
    assert len(loaded_mem.get_messages()) == len(agent_mem.get_messages())
    if loaded_mem.get_messages() and agent_mem.get_messages(): # Ensure lists are not empty
      assert loaded_mem.get_messages()[-1].content == agent_mem.get_messages()[-1].content
    
    print(f"\nSnapshot log from loaded memory ({len(loaded_mem.get_snapshot_log())} snapshots):")
    for snap_info in loaded_mem.get_snapshot_log():
        print(f"  - ID: {snap_info['id']}, Time: {snap_info['timestamp']}, Desc: {snap_info['description']}")

    # Test snapshot restoration
    snapshots = agent_mem.get_snapshot_log()
    if len(snapshots) > 2:
        snapshot_to_restore_id = snapshots[1]['id'] # Restore to the second snapshot
        print(f"\nRestoring to snapshot ID: {snapshot_to_restore_id}")
        restored_successfully = agent_mem.restore_from_snapshot(snapshot_id=snapshot_to_restore_id)
        assert restored_successfully
        print(f"Messages after restoring to snapshot '{snapshot_to_restore_id}' ({len(agent_mem.get_messages())}):")
        # The number of messages should match that snapshot's state
        original_snapshot_obj = next(s for s in agent_mem.snapshots if s.snapshot_id == snapshot_to_restore_id)
        assert len(agent_mem.get_messages()) == len(original_snapshot_obj.messages)
        for msg in agent_mem.get_messages():
            print(f"  - {msg.content[:50]}...")
    else:
        print("\nNot enough snapshots to test restoration by ID robustly.")


    # Clean up
    if json_file_path.exists():
        json_file_path.unlink()
        logger.info(f"Cleaned up {json_file_path}")

    logger.info("ConversationMemory example usage completed.")
