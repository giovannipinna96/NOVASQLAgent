# examples/sandbox_sql_sandbox_example.py

import sys
import os
import json # For schema representation

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.sandbox.sql_sandbox import SQLSandbox
    # from src.sandbox.sql_sandbox import SandboxSQLError, ConnectionConfig # Potential other components
except ImportError:
    print("Warning: Could not import SQLSandbox from src.sandbox.sql_sandbox.")
    print("Using a dummy SQLSandbox class for demonstration purposes.")

    class SQLSandbox:
        """
        Dummy SQLSandbox class for demonstration.
        This dummy version will simulate SQL execution without a real database connection.
        """
        def __init__(self, db_config, max_rows_return=1000, execution_timeout_s=30):
            self.db_config = db_config # e.g., {"type": "sqlite", "path": ":memory:"} or connection string
            self.max_rows_return = max_rows_return
            self.execution_timeout_s = execution_timeout_s
            self._connected = False
            self._dummy_schema = self._initialize_dummy_schema() # Initialize a dummy schema

            print(f"Dummy SQLSandbox initialized. Config: {self.db_config}, Max Rows: {self.max_rows_return}, Timeout: {self.execution_timeout_s}s")
            self.connect() # Try to "connect"

        def _initialize_dummy_schema(self):
            # Based on db_config, we could have different dummy schemas
            if self.db_config.get("type") == "chinook_sim": # Example for specific dummy db
                return {
                    "tables": [
                        {"name": "Artists", "columns": ["ArtistId INTEGER PRIMARY KEY", "Name NVARCHAR(120)"]},
                        {"name": "Albums", "columns": ["AlbumId INTEGER PRIMARY KEY", "Title NVARCHAR(160)", "ArtistId INTEGER"], "foreign_keys": ["FOREIGN KEY (ArtistId) REFERENCES Artists(ArtistId)"]},
                        {"name": "Tracks", "columns": ["TrackId INTEGER PRIMARY KEY", "Name NVARCHAR(200)", "AlbumId INTEGER", "Milliseconds INTEGER", "UnitPrice NUMERIC(10,2)"], "foreign_keys": ["FOREIGN KEY (AlbumId) REFERENCES Albums(AlbumId)"]}
                    ]
                }
            return { # Default generic dummy schema
                "tables": [
                    {"name": "users", "columns": ["user_id INT PRIMARY KEY", "username VARCHAR(50)", "email VARCHAR(100)", "registration_date DATE"]},
                    {"name": "products", "columns": ["product_id INT PRIMARY KEY", "product_name VARCHAR(100)", "category VARCHAR(50)", "price DECIMAL(10,2)"]},
                    {"name": "orders", "columns": ["order_id INT PRIMARY KEY", "user_id INT", "product_id INT", "quantity INT", "order_date TIMESTAMP"], "foreign_keys": ["FOREIGN KEY (user_id) REFERENCES users(user_id)", "FOREIGN KEY (product_id) REFERENCES products(product_id)"]}
                ]
            }


        def connect(self):
            """
            Dummy method to simulate connecting to the database.
            """
            print(f"\nSimulating connecting to database with config: {self.db_config}...")
            # In a real scenario, this would establish a DB connection.
            self._connected = True
            print("Successfully 'connected' to dummy database.")
            return True

        def disconnect(self):
            """
            Dummy method to simulate disconnecting from the database.
            """
            print("\nSimulating disconnecting from database...")
            self._connected = False
            print("Successfully 'disconnected' from dummy database.")

        def execute_sql(self, sql_query, read_only=True, parameters=None):
            """
            Dummy method to simulate SQL query execution.
            """
            if not self._connected:
                print("Error: Not connected to the database.")
                return {"status": "error", "message": "Not connected."}

            print(f"\nExecuting SQL (dummy): '{sql_query}'")
            print(f"Read-only: {read_only}, Parameters: {parameters}")

            # Basic security simulation for read_only mode
            is_destructive = any(kw in sql_query.upper() for kw in ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT"])
            if read_only and is_destructive:
                print("Error: Write/destructive operation attempted in read-only mode.")
                return {"status": "error", "message": "Write operation attempted in read-only mode.", "query": sql_query}

            # Simulate results based on query type
            if "SELECT" in sql_query.upper():
                # Try to guess table name for more relevant dummy data
                table_name_guess = None
                if "FROM " in sql_query.upper():
                    try:
                        table_name_guess = sql_query.upper().split("FROM ")[1].split(" ")[0].strip().lower()
                    except: pass

                dummy_cols = ["id", "name", "value"]
                dummy_rows = [{"id": 1, "name": "Dummy Result A", "value": 100}, {"id": 2, "name": "Dummy Result B", "value": 200}]

                if table_name_guess and self._dummy_schema:
                    for table_info in self._dummy_schema.get("tables", []):
                        if table_info["name"].lower() == table_name_guess:
                            dummy_cols = [col.split(" ")[0] for col in table_info["columns"]] # "col_name TYPE" -> "col_name"
                            # Generate more specific dummy rows based on col names
                            dummy_rows = [{col_name: f"dummy_{col_name}_val{i+1}" for col_name in dummy_cols} for i in range(2)]
                            break

                if len(dummy_rows) > self.max_rows_return:
                    dummy_rows = dummy_rows[:self.max_rows_return]
                    print(f"Warning: Result set truncated to {self.max_rows_return} rows.")

                result = {"status": "success", "columns": dummy_cols, "rows": dummy_rows, "row_count": len(dummy_rows)}
                print(f"Query Result (dummy): {result['row_count']} rows, Columns: {result['columns']}")
                return result
            elif is_destructive and not read_only: # INSERT, UPDATE, DELETE etc.
                affected_rows = 1 # Simulate one row affected
                result = {"status": "success", "affected_rows": affected_rows, "message": "Operation successful (dummy)."}
                print(f"Query Result (dummy): {result['message']}, Affected Rows: {affected_rows}")
                return result
            else: # Other types of SQL (e.g., CREATE TABLE, SET, etc.) or safe non-SELECTs
                result = {"status": "success", "message": "Command executed successfully (dummy)."}
                print(f"Query Result (dummy): {result['message']}")
                return result

        def get_schema_info(self, tables=None):
            """
            Dummy method to get schema information.
            'tables' is an optional list of table names to get specific schema for.
            """
            if not self._connected:
                print("Error: Not connected to the database for schema info.")
                return None

            print(f"\nRetrieving schema information (dummy) for tables: {tables if tables else 'ALL'}")
            if tables: # Filter schema if specific tables are requested
                requested_schema_tables = [t for t in self._dummy_schema.get("tables", []) if t["name"] in tables]
                if not requested_schema_tables:
                    print(f"No schema information found for specified tables: {tables}")
                    return {"tables": []}
                schema_to_return = {"tables": requested_schema_tables}
            else: # Return full dummy schema
                schema_to_return = self._dummy_schema

            # In a real scenario, this would query database metadata tables.
            print(f"Schema Info (dummy): {json.dumps(schema_to_return, indent=2)}")
            return schema_to_return

def main():
    print("--- SQLSandbox Module Example ---")

    # Configuration for the database connection
    # For a real SQLite in-memory DB: {"type": "sqlite", "path": ":memory:"}
    # For this dummy, the config just influences simulated behavior/schema.
    db_connection_config = {"type": "chinook_sim", "host": "localhost", "database": "Chinook"}

    try:
        sql_sandbox = SQLSandbox(
            db_config=db_connection_config,
            max_rows_return=50,      # Max rows to return from a SELECT
            execution_timeout_s=10   # Timeout for query execution
        )
    except NameError: # Fallback for dummy class
        sql_sandbox = SQLSandbox(db_config=db_connection_config) # Use default dummy params

    # Example 1: Get schema information (all tables)
    print("\n[Example 1: Get Full Schema Information]")
    schema = sql_sandbox.get_schema_info()
    # if schema:
    #     print(f"Retrieved schema for {len(schema.get('tables',[]))} tables.")

    # Example 2: Get schema for specific tables
    print("\n[Example 2: Get Schema for Specific Tables]")
    specific_schema = sql_sandbox.get_schema_info(tables=["Albums", "Tracks"]) # Using Chinook names
    # if specific_schema:
    #     print(f"Retrieved schema for tables: {[t['name'] for t in specific_schema.get('tables',[])]}")


    # Example 3: Execute a read-only SELECT query
    print("\n[Example 3: Execute SELECT Query (Read-Only)]")
    select_query = "SELECT Name, Milliseconds FROM Tracks WHERE AlbumId = 10 ORDER BY Name LIMIT 5;"
    select_result = sql_sandbox.execute_sql(select_query, read_only=True)
    # if select_result and select_result.get("status") == "success":
    #     print(f"SELECT query returned {select_result.get('row_count')} rows.")
    #     # for row in select_result.get("rows", []):
    #     #     print(f"  {row}")

    # Example 4: Execute a write query (e.g., INSERT) - needs read_only=False
    print("\n[Example 4: Execute INSERT Query (Not Read-Only)]")
    insert_query = "INSERT INTO Artists (Name) VALUES ('My New Favorite Band');"
    # Note: Dummy parameters are not really used by this simple dummy execute_sql
    insert_result = sql_sandbox.execute_sql(insert_query, read_only=False, parameters=["My New Favorite Band"])
    # if insert_result and insert_result.get("status") == "success":
    #     print(f"INSERT query successful. Affected rows: {insert_result.get('affected_rows', 0)}")

    # Example 5: Attempt a destructive query in read-only mode (should be blocked)
    print("\n[Example 5: Attempt DESTRUCTIVE Query in Read-Only Mode]")
    drop_query = "DROP TABLE Tracks;"
    drop_result_readonly = sql_sandbox.execute_sql(drop_query, read_only=True)
    # if drop_result_readonly and drop_result_readonly.get("status") == "error":
    #     print(f"Attempt to {drop_query} in read-only mode was blocked: {drop_result_readonly.get('message')}")

    # Example 6: Execute a destructive query with read_only=False (dummy will "allow")
    print("\n[Example 6: Execute DESTRUCTIVE Query (Not Read-Only)]")
    delete_query = "DELETE FROM Artists WHERE ArtistId > 300;" # Example destructive query
    delete_result = sql_sandbox.execute_sql(delete_query, read_only=False)
    # if delete_result and delete_result.get("status") == "success":
    #     print(f"DELETE query simulated. Affected rows: {delete_result.get('affected_rows', 0)}")


    # Example 7: Query with parameters (dummy just logs them)
    print("\n[Example 7: Query with Parameters]")
    query_with_params = "SELECT * FROM Albums WHERE ArtistId = ? AND Title LIKE ?;"
    params = (5, '%Rock%') # Example parameters for SQL placeholder
    result_params = sql_sandbox.execute_sql(query_with_params, read_only=True, parameters=params)


    # Disconnect (if the sandbox requires explicit disconnection)
    if hasattr(sql_sandbox, "disconnect"):
        sql_sandbox.disconnect()

    print("\n--- SQLSandbox Module Example Complete ---")

if __name__ == "__main__":
    main()
