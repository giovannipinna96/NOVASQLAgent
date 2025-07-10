"""
This module provides a structured interface for translating SQL queries from one dialect to another
using the open-source `sqlglot` library (https://github.com/tobymao/sqlglot).

MODULE OBJECTIVES:

1. PURPOSE:
    - The primary goal of this file is to offer a clean and extensible implementation for translating SQL queries
      across different SQL dialects using `sqlglot` as the backend parser and transpiler.
    - It allows for integration into complex pipelines where SQL dialect normalization is required,
      such as database migration, benchmarking, or SQL generation tasks.

2. FUNCTIONALITY:
    - This module should expose either a main class or a set of utility functions that:
        • Accept an SQL query as input (in string form)
        • Accept a target SQL dialect (as a string or Enum)
        • Use `sqlglot` to parse and transpile the query to the desired dialect
        • Return the translated query as a string, or raise an informative exception if translation fails

3. DIALECT ENUM:
    - A custom Enum should be defined within this file to explicitly list and constrain the supported SQL dialects.
    - This improves type safety and ensures consistency across the codebase by avoiding loose string matching.
    - Example enum values may include: `MYSQL`, `POSTGRES`, `SQLITE`, `BIGQUERY`, `DUCKDB`, etc.
    - The mapping between Enum values and `sqlglot`-recognized dialect strings must be explicitly handled.

4. ERROR HANDLING AND VALIDATION:
    - The implementation should include proper validation of inputs and dialects.
    - In case of unsupported dialects, malformed queries, or sqlglot errors, raise exceptions with clear and informative messages.

5. CODE QUALITY AND BEST PRACTICES:
    - The implementation must follow Python best practices:
        • Use PEP8 style and naming conventions
        • Annotate all functions and methods with proper type hints
        • Include docstrings for all public interfaces
        • Use logging for traceability and debugging
        • Keep the interface modular and easily extendable (e.g., pluggable dialect support or reverse-translation)

6. EXTENSIBILITY:
    - The system should be designed to allow future enhancements such as:
        • Bidirectional translation (auto-detect source dialect)
        • Query formatting and linting
        • Batch translation of multiple queries
        • Integration with SQL AST analysis if required

This module is a foundational building block for any system requiring SQL query adaptation across engines
and must ensure reliability, maintainability, and clarity in its translation pipeline.
"""
import logging
from enum import Enum
from typing import Optional

try:
    import sqlglot
    from sqlglot import Dialect, ParseError, transpile
except ImportError:
    print("sqlglot library not found. Please install it: pip install sqlglot")
    # Define dummy classes/functions if not installed for parsing, but it won't work.
    class Dialect: # type: ignore
        @classmethod
        def get(cls, key: Optional[str]) -> Optional['Dialect']: return None
    class ParseError(Exception): pass # type: ignore
    def transpile(*args, **kwargs): # type: ignore
        raise NotImplementedError("sqlglot library is not installed.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLDialect(Enum):
    """
    Custom Enum for supported SQL dialects.
    Maps user-friendly names to sqlglot's dialect strings.
    """
    BIGQUERY = "bigquery"
    CLICKHOUSE = "clickhouse"
    DATABRICKS = "databricks" # or "spark" if sqlglot prefers
    DUCKDB = "duckdb"
    HIVE = "hive"
    MYSQL = "mysql"
    ORACLE = "oracle"
    POSTGRES = "postgres"
    PRESTO = "presto"
    REDSHIFT = "redshift"
    SNOWFLAKE = "snowflake"
    SPARK = "spark"
    SQLITE = "sqlite"
    STARROCKS = "starrocks"
    TABLEAU = "tableau"
    TERADATA = "teradata"
    TRINO = "trino" # Trino was formerly PrestoSQL
    TSQL = "tsql"   # Transact-SQL (Microsoft SQL Server)

    @classmethod
    def from_string(cls, s: str) -> 'SQLDialect':
        """Converts a string to a SQLDialect enum member, case-insensitively."""
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unsupported SQL dialect string: '{s}'. Supported: {[member.name for member in cls]}")

    def to_sqlglot_dialect(self) -> str:
        """Returns the sqlglot-recognized dialect string."""
        return self.value


class SQLTranslator:
    """
    Provides functionality to translate SQL queries between different dialects
    using the sqlglot library.
    """

    def __init__(self, source_dialect: Optional[SQLDialect] = None):
        """
        Initializes the SQLTranslator.

        Args:
            source_dialect: The dialect of the input SQL queries.
                            If None, sqlglot will attempt to auto-detect it.
        """
        self.source_dialect_str: Optional[str] = source_dialect.to_sqlglot_dialect() if source_dialect else None
        if source_dialect:
            # Validate that the source dialect is known to sqlglot
            if Dialect.get(self.source_dialect_str) is None:
                logger.error(f"sqlglot does not support the source dialect: {self.source_dialect_str}")
                raise ValueError(f"Unsupported source SQL dialect for sqlglot: {self.source_dialect_str}")
        logger.info(f"SQLTranslator initialized. Source dialect: {self.source_dialect_str or 'auto-detect'}")


    def translate(
        self,
        sql_query: str,
        target_dialect: SQLDialect,
        pretty: bool = True,
        source_dialect_override: Optional[SQLDialect] = None
    ) -> str:
        """
        Translates an SQL query to the target dialect.

        Args:
            sql_query: The SQL query string to translate.
            target_dialect: The target SQLDialect enum member.
            pretty: If True, the output SQL will be formatted nicely.
            source_dialect_override: Optionally override the instance's source dialect for this specific translation.

        Returns:
            The translated SQL query as a string.

        Raises:
            ValueError: If the target dialect is unsupported or if the query is empty.
            sqlglot.ParseError: If sqlglot fails to parse the input query.
            Exception: For other sqlglot transpilation errors.
        """
        if not sql_query.strip():
            logger.error("Input SQL query is empty or contains only whitespace.")
            raise ValueError("SQL query cannot be empty.")

        target_dialect_str = target_dialect.to_sqlglot_dialect()
        if Dialect.get(target_dialect_str) is None:
            logger.error(f"sqlglot does not support the target dialect: {target_dialect_str}")
            raise ValueError(f"Unsupported target SQL dialect for sqlglot: {target_dialect_str}")

        current_source_dialect_str = self.source_dialect_str
        if source_dialect_override:
            current_source_dialect_str = source_dialect_override.to_sqlglot_dialect()
            if Dialect.get(current_source_dialect_str) is None: # type: ignore
                 logger.error(f"sqlglot does not support the overridden source dialect: {current_source_dialect_str}")
                 raise ValueError(f"Unsupported overridden source SQL dialect for sqlglot: {current_source_dialect_str}")


        logger.debug(f"Attempting to translate query from '{current_source_dialect_str or 'auto-detect'}' to '{target_dialect_str}'. Query: {sql_query[:100]}...")

        try:
            # sqlglot.transpile handles both parsing and generation.
            # `read` is for source dialect, `write` is for target.
            translated_expressions = transpile(
                sql_query,
                read=current_source_dialect_str, # None means auto-detect
                write=target_dialect_str,
                pretty=pretty,
                # Other options like `identity` (if read==write), `comments`, etc. can be added.
            )
            if not translated_expressions: # Should not happen if transpile itself doesn't error
                # This case might occur if input is only comments and comments are stripped
                logger.warning("Transpilation resulted in an empty output, possibly due to comment-only input or stripping.")
                return "" # Return empty string for empty or comment-only results
                
            # sqlglot returns a list of strings if multiple statements are transpiled.
            # We should join them or handle them as needed. For now, assume single statement or join.
            translated_query = "\n;\n".join(translated_expressions)


            logger.info(f"Successfully translated query to {target_dialect.name}. Output: {translated_query[:100]}...")
            return translated_query
        except ParseError as pe:
            logger.error(f"sqlglot parsing error for dialect '{current_source_dialect_str or 'auto-detect'}': {pe}")
            logger.error(f"Problematic SQL (first 200 chars): {sql_query[:200]}")
            # pe.errors often contains detailed error messages with context
            for err in getattr(pe, 'errors', []):
                logger.error(f"  - {err}")
            raise
        except Exception as e:
            logger.error(f"sqlglot transpilation error from '{current_source_dialect_str or 'auto-detect'}' to '{target_dialect_str}': {e}")
            logger.error(f"Problematic SQL (first 200 chars): {sql_query[:200]}")
            raise


    @staticmethod
    def parse_sql(sql_query: str, dialect: Optional[SQLDialect] = None) -> sqlglot.Expression: # type: ignore
        """
        Parses an SQL query into a sqlglot Expression (AST).
        Uses sqlglot.parse_one, so expects a single statement. For multiple, use sqlglot.parse.

        Args:
            sql_query: The SQL query string.
            dialect: The SQLDialect of the query. If None, sqlglot attempts auto-detection.

        Returns:
            A sqlglot Expression object representing the AST.

        Raises:
            sqlglot.ParseError: If parsing fails.
            ValueError: If multiple SQL statements are provided to parse_one.
        """
        dialect_str = dialect.to_sqlglot_dialect() if dialect else None
        try:
            # Check if there are multiple statements, as parse_one expects a single one.
            # A simple check; sqlglot.parse() handles multiple statements naturally.
            # For consistency with transpile potentially handling multiple, this might need adjustment
            # or users should be aware parse_sql is for single statements.
            # For now, let's stick to parse_one and document its behavior.
            parsed_expressions = sqlglot.parse(sql_query, read=dialect_str)
            if len(parsed_expressions) > 1:
                logger.warning(f"Multiple SQL statements ({len(parsed_expressions)}) found. parse_sql using parse_one expects a single statement. Returning AST for the first statement.")
            
            if not parsed_expressions or parsed_expressions[0] is None : # parse can return [None] for comment-only
                logger.error(f"Parsing resulted in no valid expression for dialect '{dialect_str or 'auto-detect'}'. Input might be empty or comments only.")
                raise ParseError("No valid SQL expression found after parsing.")


            parsed_expression = parsed_expressions[0]
            logger.info(f"Successfully parsed SQL query with dialect '{dialect_str or 'auto-detect'}'.")
            return parsed_expression
        except ParseError as pe:
            logger.error(f"sqlglot parsing error for dialect '{dialect_str or 'auto-detect'}': {pe}")
            logger.error(f"Problematic SQL (first 200 chars): {sql_query[:200]}")
            for err in getattr(pe, 'errors', []):
                logger.error(f"  - {err}")
            raise

    @staticmethod
    def render_ast(ast_expression: sqlglot.Expression, dialect: SQLDialect, pretty: bool = True) -> str: # type: ignore
        """
        Renders a sqlglot AST Expression back into an SQL query string for a specific dialect.

        Args:
            ast_expression: The sqlglot Expression (AST).
            dialect: The target SQLDialect for rendering.
            pretty: If True, format the output SQL nicely.

        Returns:
            The rendered SQL query string.
        """
        dialect_str = dialect.to_sqlglot_dialect()
        if Dialect.get(dialect_str) is None:
            raise ValueError(f"Unsupported SQL dialect for rendering: {dialect_str}")
        try:
            rendered_sql = ast_expression.sql(dialect=dialect_str, pretty=pretty)
            logger.info(f"Successfully rendered AST to SQL for dialect '{dialect_str}'.")
            return rendered_sql
        except Exception as e:
            logger.error(f"Error rendering AST to SQL for dialect '{dialect_str}': {e}")
            raise


# Example Usage
if __name__ == "__main__":
    # Ensure sqlglot is installed: pip install sqlglot
    try:
        # Example 1: Simple translation
        translator_auto_source = SQLTranslator() # Auto-detect source dialect
        
        sqlite_query = "SELECT CAST(strftime('%Y-%m-%d %H:%M:%S', my_timestamp) AS TEXT) FROM my_table;"
        
        logger.info(f"\nInput (SQLite-like): {sqlite_query}")

        try:
            # Translate to PostgreSQL
            pg_translated_query = translator_auto_source.translate(sqlite_query, SQLDialect.POSTGRES)
            logger.info(f"Translated to PostgreSQL: {pg_translated_query}")
        except Exception as e:
            logger.error(f"Error in example 1 (PG): {e}")

        try:
            # Translate to MySQL
            mysql_translated_query = translator_auto_source.translate(sqlite_query, SQLDialect.MYSQL)
            logger.info(f"Translated to MySQL: {mysql_translated_query}")
        except Exception as e:
            logger.error(f"Error in example 1 (MySQL): {e}")


        # Example 2: Explicit source dialect
        translator_sqlite_source = SQLTranslator(source_dialect=SQLDialect.SQLITE)
        sqlite_query_for_source = "SELECT STRFTIME('%Y', my_date_col) FROM dates;"
        logger.info(f"\nInput (explicit SQLite source): {sqlite_query_for_source}")
        
        try:
            duckdb_query_from_sqlite = translator_sqlite_source.translate(sqlite_query_for_source, SQLDialect.DUCKDB)
            logger.info(f"Translated from SQLite to DuckDB: {duckdb_query_from_sqlite}")
        except Exception as e:
            logger.error(f"Error in example 2 (DuckDB): {e}")

        # Example 3: Parsing and Rendering (AST manipulation placeholder)
        generic_query = "SELECT a, b FROM c WHERE d > 10 GROUP BY a ORDER BY b DESC LIMIT 5"
        logger.info(f"\nInput (generic for parsing): {generic_query}")
        try:
            ast = SQLTranslator.parse_sql(generic_query) # Auto-detect dialect for parsing
            logger.info(f"Parsed AST (type): {type(ast)}")
            
            rendered_bigquery = SQLTranslator.render_ast(ast, SQLDialect.BIGQUERY)
            logger.info(f"AST rendered to BigQuery: {rendered_bigquery}")
            
            rendered_tsql = SQLTranslator.render_ast(ast, SQLDialect.TSQL)
            logger.info(f"AST rendered to T-SQL (SQL Server): {rendered_tsql}")
        except Exception as e:
            logger.error(f"Error in example 3 (AST): {e}")


        # Example 4: Handling a more complex query with CTEs
        complex_query_duckdb = """
        WITH monthly_sales AS (
            SELECT
                STRFTIME(sale_date, '%Y-%m') AS sale_month,
                SUM(amount) AS total_sales
            FROM sales
            WHERE product_category = 'Electronics'
            GROUP BY 1
        )
        SELECT
            sale_month,
            total_sales,
            AVG(total_sales) OVER (ORDER BY sale_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_sales
        FROM monthly_sales
        ORDER BY sale_month;
        """
        translator_duckdb_source = SQLTranslator(source_dialect=SQLDialect.DUCKDB)
        logger.info(f"\nInput (complex DuckDB): {complex_query_duckdb.strip()}")

        try:
            translated_to_snowflake = translator_duckdb_source.translate(complex_query_duckdb, SQLDialect.SNOWFLAKE)
            logger.info(f"Translated from DuckDB to Snowflake:\n{translated_to_snowflake}")
        except Exception as e:
            logger.error(f"Error in example 4 (Snowflake): {e}")

        # Example 5: Error handling for invalid SQL
        invalid_sql = "SELEC * FRM table_that_does_not_exist"
        logger.info(f"\nInput (invalid SQL): {invalid_sql}")
        try:
            translator_auto_source.translate(invalid_sql, SQLDialect.SQLITE)
        except ParseError as pe:
            first_error = pe.errors[0] if pe.errors else {"description": "Unknown parse error"}
            logger.warning(f"Caught expected ParseError for invalid SQL: {first_error['description']}")
        except Exception as e:
            logger.error(f"Unexpected error for invalid SQL: {e}")

        # Example 6: Unsupported dialect string for Enum
        try:
            SQLDialect.from_string("NONEXISTENT_DIALECT")
        except ValueError as ve:
            logger.warning(f"\nCaught expected ValueError for unsupported dialect string: {ve}")
            
        # Example 7: Query with only comments
        comment_only_query = "-- This is just a comment"
        logger.info(f"\nInput (comment only): {comment_only_query}")
        try:
            translated_comment = translator_auto_source.translate(comment_only_query, SQLDialect.POSTGRES)
            logger.info(f"Translated comment-only query to PostgreSQL: '{translated_comment}' (expected empty or comment)")
            assert translated_comment == "" or translated_comment.strip().startswith("--")
        except Exception as e:
            logger.error(f"Error in example 7 (comment only): {e}")

        # Example 8: Multiple statements
        multiple_statements_sql = "SELECT * FROM table1; INSERT INTO table2 VALUES (1, 'test');"
        logger.info(f"\nInput (multiple statements): {multiple_statements_sql}")
        try:
            translated_multi = translator_auto_source.translate(multiple_statements_sql, SQLDialect.SQLITE)
            logger.info(f"Translated multiple statements to SQLite:\n{translated_multi}")
            assert "SELECT * FROM table1" in translated_multi
            assert "INSERT INTO table2 VALUES (1, 'test')" in translated_multi
        except Exception as e:
            logger.error(f"Error in example 8 (multiple statements): {e}")


    except ImportError:
        logger.critical("sqlglot library is not installed. Examples cannot run. Please run: pip install sqlglot")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in the main example block: {e}", exc_info=True)

    logger.info("\nSQLTranslator example usage finished.")
