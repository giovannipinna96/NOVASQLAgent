# examples/sandbox_filesystem_sandbox_example.py

import sys
import os

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.sandbox.filesystem_sandbox import FileSystemSandbox
    # from src.sandbox.filesystem_sandbox import SandboxException # Potential custom exception
except ImportError:
    print("Warning: Could not import FileSystemSandbox from src.sandbox.filesystem_sandbox.")
    print("Using a dummy FileSystemSandbox class for demonstration purposes.")

    class FileSystemSandbox:
        """
        Dummy FileSystemSandbox class for demonstration.
        This dummy version will simulate operations without actual file I/O
        or by using a very temporary, controlled directory if needed for realism.
        For simplicity, it mostly manipulates an in-memory dictionary representing the filesystem.
        """
        def __init__(self, base_path, max_space_mb=10, allowed_extensions=None):
            self.base_path = base_path # This would be a real path in the actual class
            self.max_space_mb = max_space_mb
            self.allowed_extensions = allowed_extensions if allowed_extensions else ["*", ".txt", ".py", ".json", ".csv", ".sql"] # Dummy allows common ones

            # In-memory representation of the sandboxed file system
            self._virtual_fs = {} # Stores path: content or path: {'type': 'directory'}
            self._virtual_fs["/"] = {'type': 'directory', 'content': {}} # Root of the sandbox

            print(f"Dummy FileSystemSandbox initialized. Base (virtual) path: '/', Max space: {max_space_mb}MB")
            print(f"Allowed extensions: {self.allowed_extensions}")
            # In a real sandbox, os.makedirs(base_path, exist_ok=True) might be called.

        def _normalize_path(self, path):
            # Simplified normalization for the virtual FS
            if not path.startswith("/"):
                path = "/" + path
            path = os.path.normpath(path)
            if path == ".": path = "/"
            return path

        def _get_node(self, path):
            parts = [p for p in path.split('/') if p]
            current_level = self._virtual_fs["/"]['content']
            node = self._virtual_fs["/"]
            for part in parts:
                if part not in current_level:
                    return None, None # Node not found, parent doesn't have it
                node = current_level[part]
                if node.get('type') == 'directory' and parts.index(part) < len(parts) -1:
                    current_level = node['content']
            return node, current_level # Returns the node itself, and its parent's content dict

        def _get_parent_dir_node(self, path):
            parts = [p for p in path.split('/') if p]
            if not parts: return self._virtual_fs["/"], self._virtual_fs # Parent of root is root itself for this logic

            current_level_content = self._virtual_fs["/"]['content']
            parent_node = self._virtual_fs["/"]

            for i, part in enumerate(parts[:-1]): # Iterate up to the parent directory
                if part not in current_level_content or current_level_content[part].get('type') != 'directory':
                    return None, None # Parent path is invalid or not a directory
                parent_node = current_level_content[part]
                current_level_content = parent_node['content']

            return parent_node, current_level_content # parent_node is the dir, current_level_content is its 'content' dict


        def create_file(self, file_path, content=""):
            """
            Dummy method to create a file in the sandbox.
            """
            norm_path = self._normalize_path(file_path)
            print(f"\nAttempting to create file (dummy): '{norm_path}'")

            file_ext = os.path.splitext(norm_path)[1]
            if "*" not in self.allowed_extensions and file_ext not in self.allowed_extensions:
                print(f"Error: File extension '{file_ext}' not allowed for '{norm_path}'.")
                return False

            # Simulate size check
            if len(content.encode('utf-8')) / (1024*1024) > self.max_space_mb: # Simplified
                print(f"Error: Content size exceeds remaining sandbox space for '{norm_path}'.")
                return False

            parent_dir_path = os.path.dirname(norm_path)
            file_name = os.path.basename(norm_path)

            parent_node, parent_content_dict = self._get_parent_dir_node(norm_path)

            if parent_node is None or parent_node.get('type') != 'directory':
                print(f"Error: Parent directory for '{norm_path}' does not exist or is not a directory.")
                return False

            if file_name in parent_content_dict and parent_content_dict[file_name].get('type') == 'directory':
                print(f"Error: Cannot create file '{norm_path}', a directory with the same name exists.")
                return False

            parent_content_dict[file_name] = {'type': 'file', 'content': content, 'size': len(content.encode('utf-8'))}
            print(f"File '{norm_path}' created successfully (dummy) with content: '{content[:30]}...'")
            return True

        def read_file(self, file_path):
            """
            Dummy method to read a file from the sandbox.
            """
            norm_path = self._normalize_path(file_path)
            print(f"\nAttempting to read file (dummy): '{norm_path}'")

            node, _ = self._get_node(norm_path)

            if node and node.get('type') == 'file':
                content = node['content']
                print(f"File '{norm_path}' content (dummy): '{content}'")
                return content
            else:
                print(f"Error: File '{norm_path}' not found or is a directory (dummy).")
                return None

        def list_directory(self, dir_path="/"):
            """
            Dummy method to list contents of a directory in the sandbox.
            """
            norm_path = self._normalize_path(dir_path)
            print(f"\nListing directory (dummy): '{norm_path}'")

            node, _ = self._get_node(norm_path)

            if node and node.get('type') == 'directory':
                items = list(node['content'].keys())
                print(f"Contents of '{norm_path}' (dummy): {items}")
                return items
            else:
                print(f"Error: Directory '{norm_path}' not found or is a file (dummy).")
                return None


        def delete_file(self, file_path):
            """
            Dummy method to delete a file in the sandbox.
            """
            norm_path = self._normalize_path(file_path)
            print(f"\nAttempting to delete file (dummy): '{norm_path}'")

            parent_node, parent_content_dict = self._get_parent_dir_node(norm_path)
            file_name = os.path.basename(norm_path)

            if parent_node and file_name in parent_content_dict:
                if parent_content_dict[file_name].get('type') == 'directory':
                    print(f"Error: '{norm_path}' is a directory. Use delete_directory (if available) or ensure it's empty.")
                    return False # Or handle recursive deletion if that's a feature
                del parent_content_dict[file_name]
                print(f"File '{norm_path}' deleted successfully (dummy).")
                return True
            else:
                print(f"Error: File '{norm_path}' not found for deletion (dummy).")
                return False

        def create_directory(self, dir_path):
            """Dummy method to create a directory."""
            norm_path = self._normalize_path(dir_path)
            print(f"\nAttempting to create directory (dummy): '{norm_path}'")
            if norm_path == "/":
                print("Cannot create root directory, it already exists.")
                return True # Or False, depending on desired strictness

            parent_dir_path = os.path.dirname(norm_path)
            dir_name = os.path.basename(norm_path)

            parent_node, parent_content_dict = self._get_parent_dir_node(norm_path)

            if parent_node is None or parent_node.get('type') != 'directory':
                print(f"Error: Parent directory for '{norm_path}' does not exist or is not a directory.")
                return False

            if dir_name in parent_content_dict:
                print(f"Error: Item '{dir_name}' already exists in '{parent_dir_path}'.")
                return False

            parent_content_dict[dir_name] = {'type': 'directory', 'content': {}}
            print(f"Directory '{norm_path}' created successfully (dummy).")
            return True


def main():
    print("--- FileSystemSandbox Module Example ---")

    # In a real scenario, base_path would be a dedicated, temporary directory.
    # For this example, the dummy class uses a virtual filesystem.
    sandbox_base_path = "/tmp/my_safe_sandbox_area" # Actual path for real class

    try:
        # You might configure allowed file types, max size, etc.
        fs_sandbox = FileSystemSandbox(
            base_path=sandbox_base_path,
            max_space_mb=1, # Small limit for testing
            allowed_extensions=[".txt", ".log", ".csv"]
        )
    except NameError: # Fallback for dummy class
        fs_sandbox = FileSystemSandbox(
            base_path=sandbox_base_path, # The dummy uses this conceptually
            max_space_mb=1,
            allowed_extensions=[".txt", ".log", ".csv"]
        )

    # Example 1: Create a directory
    print("\n[Example 1: Create Directory]")
    fs_sandbox.create_directory("/data")
    fs_sandbox.create_directory("data/reports") # Relative to sandbox root

    # Example 2: Create a file
    print("\n[Example 2: Create a File]")
    file1_path = "/data/reports/report_today.txt"
    file1_content = "This is today's report.\nStatus: OK."
    fs_sandbox.create_file(file1_path, file1_content)

    # Example 3: Attempt to create a file with a disallowed extension
    print("\n[Example 3: Create File with Disallowed Extension]")
    fs_sandbox.create_file("/data/script.py", "print('hello')") # .py might be disallowed by dummy config

    # Example 4: Read the created file
    print("\n[Example 4: Read File]")
    retrieved_content = fs_sandbox.read_file(file1_path)
    # if retrieved_content:
    #     print(f"Content of '{file1_path}':\n{retrieved_content}")

    # Example 5: List directory contents
    print("\n[Example 5: List Directory Contents]")
    print("Listing sandbox root ('/'):")
    fs_sandbox.list_directory("/")
    print("Listing '/data':")
    fs_sandbox.list_directory("/data")
    print("Listing '/data/reports':")
    fs_sandbox.list_directory("/data/reports")


    # Example 6: Try to read a non-existent file
    print("\n[Example 6: Read Non-existent File]")
    fs_sandbox.read_file("/data/non_existent.txt")

    # Example 7: Delete the file
    print("\n[Example 7: Delete File]")
    fs_sandbox.delete_file(file1_path)
    fs_sandbox.read_file(file1_path) # Attempt to read after delete

    # Example 8: Try to create a file outside the conceptual base_path (dummy may not fully enforce this)
    # A real sandbox would strictly prevent path traversal like "../../../etc/passwd"
    # The dummy's _normalize_path helps, but real checks are more robust.
    print("\n[Example 8: Attempt Path Traversal (Conceptual)]")
    fs_sandbox.create_file("../outside_file.txt", "danger!") # Dummy will likely normalize this to /outside_file.txt

    # Example 9: Create another file to test deletion of non-empty dir (if applicable)
    fs_sandbox.create_file("/data/another.txt", "temp data")
    if hasattr(fs_sandbox, "delete_directory"):
        print("\n[Example 9: Delete Directory - not implemented in this dummy]")
        # fs_sandbox.delete_directory("/data") # This would fail if /data is not empty and not recursive
    else:
        print("\n[Example 9: delete_directory not shown as not in basic dummy API]")


    print("\n--- FileSystemSandbox Module Example Complete ---")
    # In a real application, you might want a cleanup method for the sandbox directory.
    # e.g., fs_sandbox.cleanup() or shutil.rmtree(sandbox_base_path) outside the class.

if __name__ == "__main__":
    main()
