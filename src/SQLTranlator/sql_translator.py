"""
Module for translating SQL queries between dialects using sqlglot.

See: https://github.com/tobymao/sqlglot
"""

import logging
from enum import Enum
from typing import Optional

__all__ = ["SQLDialect", "SQLTranslator"]

try:
    import sqlglot
    from sqlglot import Dialect, ParseError, transpile
except ImportError:
    raise ImportError("sqlglot library is not installed. Please run: pip install sqlglot")

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SQLDialect(Enum):
    """Enum of supported SQL dialects mapped to sqlglot dialect strings."""
    BIGQUERY = "bigquery"
    CLICKHOUSE = "clickhouse"
    DATABRICKS = "databricks"
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
    TRINO = "trino"
    TSQL = "tsql"

    @classmethod
    def from_string(cls, s: str) -> 'SQLDialect':
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f"Unsupported SQL dialect string: '{s}'. "
                f"Supported: {[member.name for member in cls]}"
            )

    def to_sqlglot_dialect(self) -> str:
        return self.value


class SQLTranslator:
    """Translates SQL queries between dialects using sqlglot."""

    def __init__(self, source_dialect: Optional[SQLDialect] = None):
        self.source_dialect_str = source_dialect.to_sqlglot_dialect() if source_dialect else None

        if self.source_dialect_str and not Dialect.get(self.source_dialect_str):
            raise ValueError(f"Unsupported source dialect: {self.source_dialect_str}")

        logger.info(f"SQLTranslator initialized. Source dialect: {self.source_dialect_str or 'auto-detect'}")

    def translate(
        self,
        sql_query: str,
        target_dialect: SQLDialect,
        pretty: bool = True,
        source_dialect_override: Optional[SQLDialect] = None
    ) -> str:
        if not sql_query.strip():
            raise ValueError("SQL query cannot be empty.")

        target_dialect_str = target_dialect.to_sqlglot_dialect()
        if not Dialect.get(target_dialect_str):
            raise ValueError(f"Unsupported target dialect: {target_dialect_str}")

        source_dialect_str = source_dialect_override.to_sqlglot_dialect() if source_dialect_override else self.source_dialect_str
        if source_dialect_override and not Dialect.get(source_dialect_str):
            raise ValueError(f"Unsupported overridden source dialect: {source_dialect_str}")

        try:
            result = transpile(
                sql_query,
                read=source_dialect_str,
                write=target_dialect_str,
                pretty=pretty
            )
            return "\n;\n".join(result) if result else ""
        except ParseError as e:
            logger.error(f"Parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise

    @staticmethod
    def parse_sql(sql_query: str, dialect: Optional[SQLDialect] = None) -> sqlglot.Expression:
        dialect_str = dialect.to_sqlglot_dialect() if dialect else None
        parsed = sqlglot.parse(sql_query, read=dialect_str)

        if not parsed or parsed[0] is None:
            raise ParseError("No valid SQL expression found.")

        if len(parsed) > 1:
            logger.warning("Multiple SQL statements detected. Returning only the first.")

        return parsed[0]

    @staticmethod
    def render_ast(ast_expression: sqlglot.Expression, dialect: SQLDialect, pretty: bool = True) -> str:
        dialect_str = dialect.to_sqlglot_dialect()
        if not Dialect.get(dialect_str):
            raise ValueError(f"Unsupported rendering dialect: {dialect_str}")
        return ast_expression.sql(dialect=dialect_str, pretty=pretty)
    


if __name__ == "__main__":

    def translate_to_all_dialects(sqlite_query: str) -> None:
        """
        Translates a SQLite query to all supported SQL dialects, prints the result,
        then translates each back to SQLite and compares it with the original.
        """
        print("\n=== Translating SQLite query to all supported dialects ===\n")
        translator = SQLTranslator(source_dialect=SQLDialect.SQLITE)

        for target_dialect in SQLDialect:
            print(f"\n--- {target_dialect.name} ---")
            try:
                # Translate from SQLite to target dialect
                translated = translator.translate(sqlite_query, target_dialect=target_dialect)
                print(f"Translated to {target_dialect.name}:\n{translated}")

                # Reverse translation: Target dialect -> back to SQLite
                reverse_translator = SQLTranslator(source_dialect=target_dialect)
                reversed_query = reverse_translator.translate(translated, target_dialect=SQLDialect.SQLITE)

                # Print reversed translation
                print(f"\nBack-translated to SQLITE:\n{reversed_query}")

                # Compare (normalize both queries by removing whitespace)
                original_norm = "".join(sqlite_query.split()).lower()
                reversed_norm = "".join(reversed_query.split()).lower()

                if original_norm == reversed_norm:
                    print("✅ Round-trip translation MATCHES the original.")
                else:
                    print("❌ Round-trip translation DIFFERS from the original.")

            except Exception as e:
                print(f"Translation failed: {e}")

    # Test with various queries
    translator = SQLTranslator()

    queries = [
        ("SELECT a, COUNT(*) FROM table GROUP BY a;", SQLDialect.MYSQL),
        ("SELECT STRFTIME('%Y-%m-%d', created_at) FROM users;", SQLDialect.SQLITE),
        ("-- Just a comment", SQLDialect.POSTGRES),
        ("SELECT * FROM t1; INSERT INTO t2 VALUES (1);", SQLDialect.SQLITE),
        ("SELEC * FRM table", SQLDialect.DUCKDB)  # Invalid
    ]

    for sql, dialect in queries:
        try:
            print("\n---")
            print(f"Original SQL:\n{sql}")
            translated = translator.translate(sql, target_dialect=dialect)
            print(f"Translated SQL to {dialect.name}:\n{translated}")
        except Exception as e:
            print(f"Error: {e}")

    # AST parse/render test
    try:
        ast = SQLTranslator.parse_sql("SELECT col FROM tab WHERE x = 1")
        print("\nAST to BigQuery:\n", SQLTranslator.render_ast(ast, SQLDialect.BIGQUERY))
    except Exception as e:
        print(f"AST parsing/rendering error: {e}")

    # Invalid dialect test
    try:
        SQLDialect.from_string("nonexistent")
    except ValueError as ve:
        print(f"Caught expected dialect error: {ve}")

    print("\n=== Translating SQLite query to all dialects and back ===")
    sqlite_query = "SELECT id, name, active FROM users WHERE active = 1 AND LENGTH(name) > 3"
    translate_to_all_dialects(sqlite_query)
    print("\nTranslation round-trip testing complete.")
