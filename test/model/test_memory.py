"""
Tests for src/model/memory.py
As per instructions, this file will only contain the code, without attempting to run it
or install dependencies like 'pytest' or the actual model dependencies.
"""
import unittest
from pathlib import Path
import tempfile
import sys
import json
from datetime import datetime, timedelta

# Add src directory to sys.path
try:
    SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from model.memory import ConversationMemory, Message, MessageDirection, ConversationSnapshot, PlaceholderLLM
except ImportError as e:
    print(f"Test_Memory: Could not import ConversationMemory related classes. Error: {e}")
    ConversationMemory = None # type: ignore
    Message = None # type: ignore
    MessageDirection = None # type: ignore
    ConversationSnapshot = None # type: ignore
    PlaceholderLLM = None # type: ignore


# Mock for LLMmodel.BaseLLM if not available or for isolated testing
if PlaceholderLLM is None: # If the import from memory.py failed
    class PlaceholderLLM: # type: ignore
        def __init__(self, name: str, tokenizer: callable = None, max_tokens: int = 100):
            self.name = name
            self.tokenizer = tokenizer or (lambda text: text.split())
            self.max_tokens = max_tokens
        def summarize(self, text: str, max_length: int = 15, min_length: int = 5) -> str:
            return f"Summary: {text[:max_length]}"
        def count_tokens(self, text: str) -> int:
            return len(self.tokenizer(text))

class TestConversationMemory(unittest.TestCase):
    """ Unit tests for the ConversationMemory class. """

    def setUp(self):
        if not all([ConversationMemory, Message, MessageDirection, ConversationSnapshot, PlaceholderLLM]):
            self.skipTest("ConversationMemory or its dependencies not loaded, skipping tests.")

        self.agent_id = "test_agent_007"
        self.mock_llm = PlaceholderLLM(name="test_llm_for_memory", max_tokens=50) # Lower max_tokens for easier summary testing
        self.memory = ConversationMemory(agent_id=self.agent_id, llm_model=self.mock_llm)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test basic initialization of ConversationMemory."""
        self.assertEqual(self.memory.agent_id, self.agent_id)
        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(len(self.memory.snapshots), 1) # Initial snapshot
        self.assertEqual(self.memory.snapshots[0].description, "Initial state")
        self.assertEqual(self.memory.max_token_length, self.mock_llm.max_tokens)

    def test_add_message(self):
        """Test adding messages and snapshot creation."""
        msg_content1 = "Hello there!"
        msg_obj1 = self.memory.add_message(msg_content1, MessageDirection.INCOMING, llm_model_name="User")

        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.messages[0].content, msg_content1)
        self.assertEqual(self.memory.messages[0].direction, MessageDirection.INCOMING)
        self.assertEqual(self.memory.messages[0].llm_model_name, "User")
        self.assertIsNotNone(self.memory.messages[0].message_id)
        self.assertIsNotNone(self.memory.messages[0].timestamp)

        self.assertEqual(len(self.memory.snapshots), 2) # Initial + 1 for add
        self.assertTrue(self.memory.snapshots[1].description.startswith("Added message"))

        msg_content2 = "This is my response."
        self.memory.add_message(msg_content2, MessageDirection.OUTGOING) # LLM name should default
        self.assertEqual(len(self.memory.messages), 2)
        self.assertEqual(self.memory.messages[1].llm_model_name, self.mock_llm.name)
        self.assertEqual(len(self.memory.snapshots), 3)

    def test_get_message_by_id(self):
        """Test retrieving a message by its ID."""
        msg1 = self.memory.add_message("Content 1", MessageDirection.INCOMING)
        retrieved_msg1 = self.memory.get_message_by_id(msg1.message_id)
        self.assertEqual(retrieved_msg1, msg1)
        self.assertIsNone(self.memory.get_message_by_id("non_existent_id"))

    def test_delete_message_by_id(self):
        """Test deleting a specific message by ID."""
        msg1 = self.memory.add_message("To delete", MessageDirection.INCOMING)
        msg2 = self.memory.add_message("To keep", MessageDirection.INCOMING)
        initial_snapshots = len(self.memory.snapshots)

        self.assertTrue(self.memory.delete_message_by_id(msg1.message_id))
        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.messages[0], msg2)
        self.assertIsNone(self.memory.get_message_by_id(msg1.message_id))
        self.assertEqual(len(self.memory.snapshots), initial_snapshots + 1)
        self.assertTrue(self.memory.snapshots[-1].description.startswith("Deleted message"))

        self.assertFalse(self.memory.delete_message_by_id("non_existent_id"))
        self.assertEqual(len(self.memory.snapshots), initial_snapshots + 1) # No new snapshot for failed delete

    def test_delete_messages_before_timestamp(self):
        """Test deleting messages before a certain timestamp."""
        # Timestamps are strings, ensure comparison works. ISO format helps.
        msg1_time = datetime.utcnow() - timedelta(hours=2)
        msg2_time = datetime.utcnow() - timedelta(hours=1)
        msg3_time = datetime.utcnow()

        # Manually create messages with specific timestamps for testing
        m1 = Message(content="Old msg", direction=MessageDirection.INCOMING, timestamp=msg1_time.isoformat())
        m2 = Message(content="Mid msg", direction=MessageDirection.INCOMING, timestamp=msg2_time.isoformat())
        m3 = Message(content="New msg", direction=MessageDirection.INCOMING, timestamp=msg3_time.isoformat())
        self.memory.messages = [m1, m2, m3]
        self.memory._create_snapshot("Added 3 manual messages") # type: ignore # private method access for test setup
        initial_snapshots = len(self.memory.snapshots)

        cutoff_timestamp = (datetime.utcnow() - timedelta(minutes=90)).isoformat() # Between m1 and m2
        deleted_count = self.memory.delete_messages_before_timestamp(cutoff_timestamp)

        self.assertEqual(deleted_count, 1)
        self.assertEqual(len(self.memory.messages), 2)
        self.assertNotIn(m1, self.memory.messages)
        self.assertIn(m2, self.memory.messages)
        self.assertIn(m3, self.memory.messages)
        self.assertEqual(len(self.memory.snapshots), initial_snapshots + 1)

    def test_delete_messages_by_filter(self):
        """Test deleting messages using a custom filter function."""
        self.memory.add_message("Message A user", MessageDirection.INCOMING, llm_model_name="User")
        self.memory.add_message("Message B agent", MessageDirection.OUTGOING, llm_model_name=self.mock_llm.name)
        self.memory.add_message("Message C user", MessageDirection.INCOMING, llm_model_name="User")
        initial_snapshots = len(self.memory.snapshots)

        # Delete all messages from "User"
        filter_func = lambda msg: msg.llm_model_name == "User"
        deleted_count = self.memory.delete_messages_by_filter(filter_func)

        self.assertEqual(deleted_count, 2)
        self.assertEqual(len(self.memory.messages), 1)
        self.assertEqual(self.memory.messages[0].content, "Message B agent")
        self.assertEqual(len(self.memory.snapshots), initial_snapshots + 1)

    def test_conversation_summarization(self):
        """Test automatic conversation summarization when token limit is exceeded."""
        # mock_llm has max_tokens=50. tokenizer splits by space.
        # Add messages to exceed this.
        for i in range(15): # Each message ~5-7 "tokens" by dummy tokenizer
            self.memory.add_message(f"This is message number {i} in a long sequence of test inputs.", MessageDirection.INCOMING)

        # After adding enough messages, summarization should have been triggered.
        self.assertEqual(len(self.memory.messages), 1) # Should be reduced to 1 summary message
        summary_message = self.memory.messages[0]
        self.assertTrue(summary_message.content.startswith("Summary of prior conversation: Summary: This is message number 0"))
        self.assertTrue(summary_message.metadata.get("is_summary"))
        self.assertGreaterEqual(summary_message.metadata.get("original_message_count", 0), 15)
        self.assertTrue(self.memory.snapshots[-1].description == "Conversation summarized")

    def test_json_export_import(self):
        """Test exporting to JSON and importing back."""
        self.memory.add_message("Export test msg 1", MessageDirection.INCOMING)
        self.memory.add_message("Export test msg 2", MessageDirection.OUTGOING, metadata={"custom_key": "value"})

        json_file_path = self.temp_path / "test_memory.json"
        self.memory.to_json(json_file_path)
        self.assertTrue(json_file_path.exists())

        # Create a new memory instance and load from JSON
        new_llm = PlaceholderLLM(name="loaded_llm")
        loaded_memory = ConversationMemory.from_json(json_file_path, llm_model=new_llm)

        self.assertEqual(loaded_memory.agent_id, self.memory.agent_id)
        self.assertEqual(len(loaded_memory.messages), len(self.memory.messages))
        self.assertEqual(loaded_memory.messages[0].content, self.memory.messages[0].content)
        self.assertEqual(loaded_memory.messages[1].metadata.get("custom_key"), "value")
        self.assertEqual(loaded_memory.max_token_length, self.memory.max_token_length)

        # Check snapshots (basic check for count and last description)
        self.assertEqual(len(loaded_memory.snapshots), len(self.memory.snapshots))
        if self.memory.snapshots: # Ensure there are snapshots to check
             self.assertEqual(loaded_memory.snapshots[-1].description, self.memory.snapshots[-1].description)


    def test_temporal_snapshot_storage_and_restoration(self):
        """Test snapshot creation and restoration."""
        m1 = self.memory.add_message("State 1", MessageDirection.INCOMING)
        snapshot1_id = self.memory.snapshots[-1].snapshot_id
        snapshot1_time = self.memory.snapshots[-1].timestamp
        num_messages_at_snap1 = len(self.memory.messages)


        self.memory.add_message("State 2", MessageDirection.OUTGOING)
        m3 = self.memory.add_message("State 3", MessageDirection.INCOMING)
        snapshot3_id = self.memory.snapshots[-1].snapshot_id # Snapshot after adding m3

        # Restore to snapshot after m1 was added
        self.assertTrue(self.memory.restore_from_snapshot(snapshot_id=snapshot1_id))
        self.assertEqual(len(self.memory.messages), num_messages_at_snap1)
        self.assertEqual(self.memory.messages[0].content, "State 1")
        self.assertTrue(self.memory.snapshots[-1].description.startswith("Restored from snapshot"))

        # Restore again to snapshot after m3 (latest before this restore)
        # This tests restoring to a snapshot that isn't the immediately previous one in the log.
        self.assertTrue(self.memory.restore_from_snapshot(snapshot_id=snapshot3_id))
        self.assertEqual(len(self.memory.messages), 3) # m1, state2, m3
        self.assertEqual(self.memory.messages[-1].content, "State 3")

        # Test restore by timestamp (approximate)
        # Find a timestamp between snapshot1 and snapshot3
        # For precise test, need to control time, using snapshot IDs is more robust here.
        # This is a conceptual test for timestamp restore
        if len(self.memory.snapshots) > 2:
            time_to_restore_to = self.memory.snapshots[1].timestamp # e.g. timestamp of first snapshot
            self.assertTrue(self.memory.restore_from_snapshot(timestamp_iso=time_to_restore_to))
            # Check if messages match the state of that snapshot
            target_snapshot = next(s for s in self.memory.snapshots if s.timestamp == time_to_restore_to)
            self.assertEqual(len(self.memory.messages), len(target_snapshot.messages))


    def test_get_snapshot_log(self):
        """Test retrieving the snapshot log."""
        self.memory.add_message("Msg A", MessageDirection.INCOMING)
        self.memory.add_message("Msg B", MessageDirection.OUTGOING)
        log = self.memory.get_snapshot_log()
        self.assertEqual(len(log), 3) # Initial + 2 messages
        self.assertTrue(all("id" in item and "timestamp" in item and "description" in item for item in log))
        self.assertEqual(log[0]["description"], "Initial state")

    def test_empty_summarization(self):
        """Test summarization when there are no messages."""
        self.memory.messages = [] # Clear any initial messages if setUp changes
        self.memory._create_snapshot("Cleared for empty summary test") # type: ignore
        initial_snapshots = len(self.memory.snapshots)

        self.memory.summarize_conversation() # Should not fail
        self.assertEqual(len(self.memory.messages), 0)
        self.assertEqual(len(self.memory.snapshots), initial_snapshots) # No snapshot if no change


if __name__ == "__main__":
    if ConversationMemory is not None:
        print("Running ConversationMemory tests (illustrative execution)...")
        unittest.main(verbosity=2)
    else:
        print("Skipping ConversationMemory tests as ConversationMemory module could not be imported.")
        print("This is expected if dependencies are not installed or src path is incorrect in a non-execution environment.")
