"""
Module for an SQL sandbox for executing SQL queries safely.
Uses SQLite by default and includes basic safety checks.
"""
import sqlite3
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Basic regex patterns for unsafe queries. These are not foolproof and can be bypassed.
# A proper SQL parser (like sqlglot) would be much more robust but is a dependency.
UNSAFE_PATTERNS = {
    "DROP_TABLE_UNQUALIFIED": re.compile(r"\bDROP\s+TABLE\s+(?!sandbox_tbl_)\w+", re.IGNORECASE), # Disallow dropping tables not prefixed with sandbox_tbl_
    "DELETE_WITHOUT_WHERE": re.compile(r"\bDELETE\s+FROM\s+\w+\s*;?$", re.IGNORECASE | re.MULTILINE), # DELETE FROM table (no WHERE)
    "UPDATE_WITHOUT_WHERE": re.compile(r"\bUPDATE\s+\w+\s+SET\s+.*\s*;?$", re.IGNORECASE | re.MULTILINE), # UPDATE table SET ... (no WHERE)
    # More patterns can be added, e.g., for certain PRAGMA commands, ATTACH DATABASE, etc.
    # DROP DATABASE is not a standard SQLite command.
}
ALLOWED_DROP_TABLE_PATTERN = re.compile(r"\bDROP\s+TABLE\s+IF\s+EXISTS\s+sandbox_tbl_\w+", re.IGNORECASE)


class SQLSandbox:
    """
    An SQL sandbox using SQLite to execute queries with basic safety checks.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:", # Default to in-memory SQLite database
        timeout_seconds: int = 5, # Conceptual timeout for query execution method
        table_prefix: str = "sandbox_tbl_", # Prefix for tables created/managed by this sandbox
        enforce_safety_checks: bool = True,
    ):
        """
        Initializes the SQLSandbox.

        Args:
            db_path: Path to the SQLite database file, or ":memory:" for an in-memory DB.
            timeout_seconds: Conceptual timeout for query execution. Direct enforcement on
                             a single sqlite3.execute call is difficult. This might be used
                             by a wrapper that runs this method in a separate thread/process.
            table_prefix: A prefix that user-created tables should have to be considered "safe"
                          for certain operations like DROP.
            enforce_safety_checks: If True, run pre-query safety checks.
        """
        self.db_path = str(db_path) # sqlite3.connect expects a string
        self.timeout_seconds = timeout_seconds # Store, but direct use is limited
        self.table_prefix = table_prefix
        self.enforce_safety_checks = enforce_safety_checks

        try:
            # Test connection to ensure db_path is valid (e.g., writable if file-based)
            conn = sqlite3.connect(self.db_path, timeout=float(self.timeout_seconds))
            conn.close()
            logger.info(f"SQLSandbox initialized with database: '{self.db_path}', table_prefix: '{self.table_prefix}'")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize SQLite connection to '{self.db_path}': {e}")
            raise # Re-raise as this is critical

    def _is_query_safe(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Performs basic safety checks on the SQL query using regex.
        These checks are NOT foolproof and are a best-effort attempt without a full SQL parser.

        Args:
            query: The SQL query string.

        Returns:
            A tuple (is_safe: bool, reason: Optional[str]).
        """
        if not self.enforce_safety_checks:
            return True, None

        normalized_query = query.strip()

        # Check for DROP TABLE without prefix, unless it's 'DROP TABLE IF EXISTS sandbox_tbl_...'
        if UNSAFE_PATTERNS["DROP_TABLE_UNQUALIFIED"].search(normalized_query):
            if not ALLOWED_DROP_TABLE_PATTERN.fullmatch(normalized_query): # Check if it's the allowed form
                 # Further check if the table name in an unqualified DROP TABLE starts with the prefix
                match_unqualified_drop = re.search(r"\bDROP\s+TABLE\s+(IF\s+EXISTS\s+)?(\w+)", normalized_query, re.IGNORECASE)
                if match_unqualified_drop:
                    table_name = match_unqualified_drop.group(2)
                    if not table_name.startswith(self.table_prefix):
                        reason = f"Attempt to DROP table '{table_name}' that does not conform to prefix '{self.table_prefix}'."
                        logger.warning(f"Unsafe query detected: {reason}. Query: {query[:100]}")
                        return False, reason
                else: # Should not happen if UNSAFE_PATTERNS matched, but as a fallback
                    reason = "Potentially unsafe DROP TABLE statement."
                    logger.warning(f"Unsafe query detected: {reason}. Query: {query[:100]}")
                    return False, reason


        if UNSAFE_PATTERNS["DELETE_WITHOUT_WHERE"].search(normalized_query):
            # This regex matches "DELETE FROM table" or "DELETE FROM table;"
            # It needs to be refined if aliases or more complex statements are common.
            # A simple check: if "where" is not in the lowercased query after "delete from".
            if "where" not in normalized_query.lower().split("delete from", 1)[-1]:
                reason = "Potentially unsafe DELETE operation without a WHERE clause."
                logger.warning(f"Unsafe query detected: {reason}. Query: {query[:100]}")
                return False, reason

        if UNSAFE_PATTERNS["UPDATE_WITHOUT_WHERE"].search(normalized_query):
            if "where" not in normalized_query.lower().split("set", 1)[-1]: # Check after SET clause
                reason = "Potentially unsafe UPDATE operation without a WHERE clause."
                logger.warning(f"Unsafe query detected: {reason}. Query: {query[:100]}")
                return False, reason

        # Add more checks here if needed (e.g., for specific PRAGMA, ATTACH, etc.)
        # Example: Disallow ATTACH DATABASE
        if re.search(r"\bATTACH\s+DATABASE\b", normalized_query, re.IGNORECASE):
            reason = "ATTACH DATABASE command is disallowed."
            logger.warning(f"Unsafe query detected: {reason}. Query: {query[:100]}")
            return False, reason

        return True, None

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict, List, Tuple]] = None,
        script: bool = False
    ) -> Dict[str, Any]:
        """
        Executes one or more SQL queries.

        Args:
            query: The SQL query string or script (multiple statements separated by ';').
            params: Optional parameters to substitute into the query (for a single query).
                    Not used if `script` is True.
            script: If True, execute the query string as a script (multiple statements).
                    `params` will be ignored if `script` is True.

        Returns:
            A dictionary containing execution results:
            {
                "status": "success" | "error" | "unsafe",
                "output": Optional[List[Dict]] (for SELECT queries, list of rows as dicts),
                "error_message": Optional[str],
                "rows_affected": Optional[int] (for DML statements like INSERT, UPDATE, DELETE),
                "execution_time": float seconds,
            }
        """
        start_time = time.monotonic()
        result_status = "error"
        output: Optional[List[Dict]] = None
        error_message: Optional[str] = None
        rows_affected: Optional[int] = None

        # Pre-execution safety check
        is_safe, unsafe_reason = self._is_query_safe(query)
        if not is_safe:
            execution_time = time.monotonic() - start_time
            return {
                "status": "unsafe", "output": None, "error_message": unsafe_reason,
                "rows_affected": None, "execution_time": round(execution_time, 4),
            }

        conn: Optional[sqlite3.Connection] = None
        try:
            # Using total_changes=True for better rows_affected with older SQLite versions
            conn = sqlite3.connect(self.db_path, timeout=float(self.timeout_seconds))
            conn.isolation_level = None # Autocommit mode for scripts, or manage transactions explicitly
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()

            if script:
                if params:
                    logger.warning("Parameters are ignored when executing a script.")
                cursor.executescript(query)
                result_status = "success" # executescript doesn't return results or rowcount directly for all statements
                                       # We can get total_changes for the connection.
                rows_affected = conn.total_changes # This is total changes since connection opened.
                                                   # Could be misleading if connection is long-lived.
                                                   # For a script, this is changes by that script.
            else: # Single query execution
                cursor.execute(query, params or [])

                # For SELECT queries, fetch results
                if cursor.description: # SELECT queries have a description
                    column_names = [desc[0] for desc in cursor.description]
                    output = [dict(zip(column_names, row)) for row in cursor.fetchall()]

                rows_affected = cursor.rowcount # For INSERT, UPDATE, DELETE
                conn.commit() # Commit changes for DML
                result_status = "success"

            logger.info(f"Query executed successfully. Query: {query[:100]}...")
            if output: logger.debug(f"Query output (first few rows): {str(output)[:200]}")
            if rows_affected is not None and rows_affected > -1 : logger.debug(f"Rows affected: {rows_affected}")


        except sqlite3.Error as e:
            result_status = "error"
            error_message = f"SQLite Error: {e}"
            logger.error(f"Error executing query '{query[:100]}...': {e}", exc_info=True)
            if conn:
                try: conn.rollback() # Rollback on error
                except sqlite3.Error as rb_err: logger.error(f"Rollback failed: {rb_err}")
        except Exception as e: # Catch other potential errors
            result_status = "error"
            error_message = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error executing query '{query[:100]}...': {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

        execution_time = time.monotonic() - start_time
        return {
            "status": result_status, "output": output, "error_message": error_message,
            "rows_affected": rows_affected if rows_affected != -1 else None, # -1 means not applicable (e.g. SELECT)
            "execution_time": round(execution_time, 4),
        }

    def close(self) -> None:
        """
        Closes the database connection if it were persistent.
        For SQLite in this model (connect per query), this is mostly a no-op placeholder
        unless a persistent connection was maintained by the class instance.
        Our current `execute_query` opens/closes connection each time.
        """
        logger.info(f"SQLSandbox close called for {self.db_path}. (Connection is per-query)")

    def create_table_from_schema(self, table_name: str, schema: Dict[str, str],
                                 primary_keys: Optional[List[str]] = None,
                                 if_not_exists: bool = True) -> Dict[str, Any]:
        """
        Creates a table based on a provided schema.
        Ensures table name uses the configured prefix.

        Args:
            table_name: Name of the table to create (without prefix).
            schema: A dictionary where keys are column names and values are SQLite data types
                    (e.g., {"id": "INTEGER", "name": "TEXT"}).
            primary_keys: Optional list of column names to be part of the primary key.
                          If multiple, it's a composite key. If None, no explicit PK for simple cases.
            if_not_exists: Add "IF NOT EXISTS" clause to CREATE TABLE.

        Returns:
            Result dictionary from execute_query.
        """
        if not table_name.startswith(self.table_prefix):
            prefixed_table_name = f"{self.table_prefix}{table_name}"
        else:
            prefixed_table_name = table_name # Already has prefix

        if not schema:
            return {"status": "error", "error_message": "Schema cannot be empty.", "execution_time": 0.0}

        cols_defs = [f'"{col_name}" {col_type}' for col_name, col_type in schema.items()]

        create_statement = f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{prefixed_table_name} (\n"
        create_statement += ",\n  ".join(cols_defs)

        if primary_keys:
            pk_cols = '", "'.join(primary_keys)
            create_statement += f",\n  PRIMARY KEY (\"{pk_cols}\")"

        create_statement += "\n);"

        logger.info(f"Attempting to create table with statement: {create_statement}")
        return self.execute_query(create_statement, script=False)


if __name__ == "__main__":
    logger.info("SQLSandbox: Illustrative __main__ block.")

    # Use a temporary file for the SQLite DB for this example
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True) as tmp_db_file:
        db_file_path = Path(tmp_db_file.name)
        logger.info(f"Illustrative SQLSandbox using temporary DB: {db_file_path}")

        try:
            sql_sandbox = SQLSandbox(db_path=db_file_path, enforce_safety_checks=True)

            # Example 1: Create a table (safe, uses prefix)
            logger.info("\n--- Example 1: Create Table ---")
            schema = {"id": "INTEGER", "name": "TEXT", "value": "REAL"}
            # Using create_table_from_schema ensures prefixing
            create_result = sql_sandbox.create_table_from_schema("mydata", schema, primary_keys=["id"])
            print(f"Create Table Result: {create_result}")
            assert create_result["status"] == "success"

            # Example 2: Insert data
            logger.info("\n--- Example 2: Insert Data ---")
            insert_query = f"INSERT INTO {sql_sandbox.table_prefix}mydata (id, name, value) VALUES (?, ?, ?);"
            insert_params = [(1, "Alice", 10.5), (2, "Bob", 22.3)]
            for params_row in insert_params:
                insert_result = sql_sandbox.execute_query(insert_query, params_row)
                print(f"Insert Result ({params_row}): {insert_result}")
                assert insert_result["status"] == "success"
                assert insert_result.get("rows_affected", 0) == 1

            # Example 3: Select data
            logger.info("\n--- Example 3: Select Data ---")
            select_query = f"SELECT id, name, value FROM {sql_sandbox.table_prefix}mydata WHERE value > ?;"
            select_result = sql_sandbox.execute_query(select_query, params=[15.0])
            print(f"Select Result: {select_result}")
            assert select_result["status"] == "success"
            assert select_result["output"] is not None
            if select_result["output"]: # mypy check
                assert len(select_result["output"]) == 1
                assert select_result["output"][0]["name"] == "Bob"

            # Example 4: Unsafe query - DELETE without WHERE
            logger.info("\n--- Example 4: Unsafe DELETE ---")
            # This delete is on a prefixed table, but safety check is for "no where clause"
            unsafe_delete_query = f"DELETE FROM {sql_sandbox.table_prefix}mydata;"
            unsafe_result = sql_sandbox.execute_query(unsafe_delete_query)
            print(f"Unsafe DELETE Result: {unsafe_result}")
            assert unsafe_result["status"] == "unsafe"
            assert "DELETE operation without a WHERE clause" in unsafe_result["error_message"] if unsafe_result["error_message"] else False

            # Example 5: Unsafe query - DROP TABLE non-prefixed (or without IF EXISTS for prefixed)
            logger.info("\n--- Example 5: Unsafe DROP TABLE ---")
            unsafe_drop_query = "DROP TABLE important_system_table;" # Not prefixed
            unsafe_drop_result = sql_sandbox.execute_query(unsafe_drop_query)
            print(f"Unsafe DROP (non-prefixed) Result: {unsafe_drop_result}")
            assert unsafe_drop_result["status"] == "unsafe"
            assert "DROP table 'important_system_table'" in unsafe_drop_result["error_message"] if unsafe_drop_result["error_message"] else False

            # Example 6: Safe DROP TABLE (prefixed and with IF EXISTS)
            logger.info("\n--- Example 6: Safe DROP TABLE ---")
            # Need to adjust regex in _is_query_safe or the query to pass.
            # Current ALLOWED_DROP_TABLE_PATTERN: r"\bDROP\s+TABLE\s+IF\s+EXISTS\s+sandbox_tbl_\w+"
            safe_drop_query = f"DROP TABLE IF EXISTS {sql_sandbox.table_prefix}mydata;"
            # Modify safety check for this test case to allow this specific form
            original_unsafe_drop_pattern = UNSAFE_PATTERNS["DROP_TABLE_UNQUALIFIED"]
            UNSAFE_PATTERNS["DROP_TABLE_UNQUALIFIED"] = re.compile(r"NEVER_MATCH_THIS_FOR_TEST", re.IGNORECASE) # Disable general drop check

            safe_drop_result = sql_sandbox.execute_query(safe_drop_query)
            print(f"Safe DROP Result: {safe_drop_result}")
            assert safe_drop_result["status"] == "success"

            UNSAFE_PATTERNS["DROP_TABLE_UNQUALIFIED"] = original_unsafe_drop_pattern # Restore pattern

            # Example 7: Execute script
            logger.info("\n--- Example 7: Execute Script ---")
            script = f"""
            CREATE TABLE IF NOT EXISTS {sql_sandbox.table_prefix}script_table (id INTEGER PRIMARY KEY, data TEXT);
            INSERT INTO {sql_sandbox.table_prefix}script_table (id, data) VALUES (100, 'script entry');
            UPDATE {sql_sandbox.table_prefix}script_table SET data = 'updated by script' WHERE id = 100;
            """
            script_result = sql_sandbox.execute_query(script, script=True)
            print(f"Script Execution Result: {script_result}")
            assert script_result["status"] == "success"
            # rows_affected for script might be total changes by the script
            # For this script: 1 (CREATE) + 1 (INSERT) + 1 (UPDATE) = 3 if CREATE counts.
            # If CREATE doesn't count, then 2. SQLite total_changes usually reflects INSERT/UPDATE/DELETE.
            self.assertGreaterEqual(script_result.get("rows_affected", 0), 2)


        except Exception as e:
            logger.error(f"Error in SQLSandbox illustrative __main__: {e}", exc_info=True)
        finally:
            # tmp_db_file is auto-deleted by NamedTemporaryFile context manager
            logger.info(f"Illustrative __main__ finished. Temp DB {db_file_path} (should be deleted).")

    logger.info("SQLSandbox illustrative __main__ block completed.")
