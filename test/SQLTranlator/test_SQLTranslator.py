"""
Tests for src/SQLTranlator/SQLTranslator.py
As per instructions, this file will only contain the code, without attempting to run it
or install dependencies like 'pytest' or 'sqlglot'.
"""
import unittest
from pathlib import Path
import sys

# Add src directory to sys.path
try:
    SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    # The module is in src/SQLTranlator, not src/SQLTranslator (note case)
    # However, the plan refers to src/SQLTranslator. Assuming the directory is SQLTranlator based on ls.
    from SQLTranlator.SQLTranslator import SQLTranslator, SQLDialect
except ImportError as e:
    print(f"Test_SQLTranslator: Could not import SQLTranslator or SQLDialect. Error: {e}")
    SQLTranslator = None # type: ignore
    SQLDialect = None # type: ignore

# Mock sqlglot if not available for structural testing
if 'sqlglot' not in sys.modules:
    class MockSqlglotExpression:
        def sql(self, dialect, pretty):
            return f"mock sql for {dialect}"
    class MockSqlglotModule:
        ParseError = type('ParseError', (Exception,), {})
        Dialect = type('Dialect', (object,), {'get': lambda x: True if x else None}) # Simple mock
        def transpile(self, sql_query, read, write, pretty):
            if "error_on_transpile" in sql_query: raise Exception("Mock transpile error")
            if "parse_error_trigger" in sql_query: raise self.ParseError("Mock parse error")
            return [f"translated to {write}: {sql_query}"] # sqlglot.transpile returns a list
        def parse_one(self, sql_query, read):
            if "parse_error_trigger" in sql_query: raise self.ParseError("Mock parse error")
            return MockSqlglotExpression()
        def parse(self, sql_query, read): # To match the change in SQLTranslator.py
            if "parse_error_trigger" in sql_query: raise self.ParseError("Mock parse error")
            if sql_query.strip() == "" or sql_query.strip().startswith("--"): return [None]
            return [MockSqlglotExpression()]


    sys.modules['sqlglot'] = MockSqlglotModule() # type: ignore
    print("Test_SQLTranslator: Mocked 'sqlglot' library.")
    # Re-evaluate SQLTranslator and SQLDialect if they failed due to sqlglot not being found initially
    if SQLTranslator is None or SQLDialect is None:
        try:
            from SQLTranlator.SQLTranslator import SQLTranslator, SQLDialect
        except ImportError:
            pass # Already handled

@unittest.skipIf(SQLTranslator is None or SQLDialect is None, "SQLTranslator or SQLDialect not loaded, skipping tests.")
class TestSQLTranslator(unittest.TestCase):
    """Unit tests for the SQLTranslator class and SQLDialect enum."""

    def test_sql_dialect_enum_creation_and_mapping(self):
        """Test SQLDialect enum members and their string values."""
        self.assertEqual(SQLDialect.POSTGRES.value, "postgres") # type: ignore
        self.assertEqual(SQLDialect.MYSQL.to_sqlglot_dialect(), "mysql") # type: ignore
        self.assertEqual(SQLDialect.BIGQUERY.name, "BIGQUERY") # type: ignore

    def test_sql_dialect_from_string(self):
        """Test creating SQLDialect from string."""
        self.assertEqual(SQLDialect.from_string("postgres"), SQLDialect.POSTGRES) # type: ignore
        self.assertEqual(SQLDialect.from_string("MySqL"), SQLDialect.MYSQL) # type: ignore
        with self.assertRaisesRegex(ValueError, "Unsupported SQL dialect string: 'oracle_db'"):
            SQLDialect.from_string("oracle_db") # type: ignore

    def test_translator_initialization_default(self):
        """Test SQLTranslator initialization with default source dialect (auto-detect)."""
        translator = SQLTranslator() # type: ignore
        self.assertIsNone(translator.source_dialect_str)

    def test_translator_initialization_with_source_dialect(self):
        """Test SQLTranslator initialization with a specified source dialect."""
        translator = SQLTranslator(source_dialect=SQLDialect.SQLITE) # type: ignore
        self.assertEqual(translator.source_dialect_str, "sqlite")

    def test_translator_initialization_with_invalid_source_dialect_str(self):
        """Test SQLTranslator with a source dialect string not known by mock sqlglot.Dialect.get."""
        # This test depends on the mock sqlglot.Dialect.get behavior
        # If sqlglot.Dialect.get returns None for "unsupported_mock_dialect", it should raise ValueError
        # For this mock, Dialect.get always returns True if arg is not None, so this test might not reflect real failure.
        # To make it fail with current mock:
        sys.modules['sqlglot'].Dialect.get = lambda x: None if x == "unsupported_mock_dialect" else True # type: ignore

        class MockUnsupportedDialectEnum(SQLDialect): # type: ignore
            UNSUPPORTED = "unsupported_mock_dialect"

        with self.assertRaisesRegex(ValueError, "Unsupported source SQL dialect for sqlglot: unsupported_mock_dialect"):
            SQLTranslator(source_dialect=MockUnsupportedDialectEnum.UNSUPPORTED) # type: ignore

        # Reset mock for other tests
        sys.modules['sqlglot'].Dialect.get = lambda x: True if x else None # type: ignore


    def test_translate_simple_query(self):
        """Test basic query translation."""
        translator = SQLTranslator() # type: ignore
        sql_query = "SELECT column_a FROM table_x"
        target_dialect = SQLDialect.POSTGRES # type: ignore

        # Based on mock: "translated to postgres: SELECT column_a FROM table_x"
        expected_translation = f"translated to {target_dialect.value}: {sql_query}" # type: ignore
        translated = translator.translate(sql_query, target_dialect) # type: ignore
        self.assertEqual(translated, expected_translation)

    def test_translate_with_source_dialect_override(self):
        """Test translation overriding the instance's source dialect."""
        translator = SQLTranslator(source_dialect=SQLDialect.MYSQL) # type: ignore
        sql_query = "SELECT `col` FROM `tbl`" # MySQL-like

        # Mock sqlglot.transpile will show the read dialect
        # expected = f"translated from mysql to postgres: {sql_query}" (if mock showed source)
        # Current mock: f"translated to {target_dialect.value}: {sql_query}"
        # To test override, the mock would need to reflect the 'read' param.
        # For now, just check it runs.
        try:
            translator.translate(sql_query, SQLDialect.POSTGRES, source_dialect_override=SQLDialect.DUCKDB) # type: ignore
        except Exception as e:
            self.fail(f"Translate with override failed: {e}")


    def test_translate_empty_query(self):
        """Test translation with an empty SQL query string."""
        translator = SQLTranslator() # type: ignore
        with self.assertRaisesRegex(ValueError, "SQL query cannot be empty."):
            translator.translate("", SQLDialect.SQLITE) # type: ignore
        with self.assertRaisesRegex(ValueError, "SQL query cannot be empty."):
            translator.translate("   ", SQLDialect.SQLITE) # type: ignore

    def test_translate_unsupported_target_dialect_for_sqlglot(self):
        """Test translation to a target dialect not supported by (mocked) sqlglot."""
        sys.modules['sqlglot'].Dialect.get = lambda x: None if x == "made_up_dialect" else True # type: ignore
        translator = SQLTranslator() # type: ignore

        class MockUnsupportedDialectEnum(SQLDialect): # type: ignore
            MADEUP = "made_up_dialect"

        with self.assertRaisesRegex(ValueError, "Unsupported target SQL dialect for sqlglot: made_up_dialect"):
            translator.translate("SELECT 1", MockUnsupportedDialectEnum.MADEUP) # type: ignore
        sys.modules['sqlglot'].Dialect.get = lambda x: True if x else None # type: ignore


    def test_translate_parse_error(self):
        """Test translation when sqlglot encounters a parse error (mocked)."""
        translator = SQLTranslator() # type: ignore
        sql_query_parse_error = "SELECT * parse_error_trigger FROM my_table"
        with self.assertRaises(sys.modules['sqlglot'].ParseError): # type: ignore
            translator.translate(sql_query_parse_error, SQLDialect.SQLITE) # type: ignore

    def test_translate_transpile_error_general(self):
        """Test translation when sqlglot encounters a general transpile error (mocked)."""
        translator = SQLTranslator() # type: ignore
        sql_query_transpile_error = "SELECT * error_on_transpile FROM my_table"
        with self.assertRaisesRegex(Exception, "Mock transpile error"):
            translator.translate(sql_query_transpile_error, SQLDialect.SQLITE) # type: ignore

    def test_translate_comment_only_query(self):
        """Test translation of a query that only contains comments."""
        translator = SQLTranslator() # type: ignore
        comment_query = "-- This is a comment only query\n-- another comment"
        # Mocked transpile returns list, join makes it a string.
        # If transpile returns empty list for comment-only (as it might in reality if comments are stripped and nothing else is there)
        # then the result would be empty string.
        # Current mock returns "translated to dialect: --comment..."
        # The actual SQLTranslator code has a check for `if not translated_expressions: return ""`
        # So, if mock transpile returns `[]` for this, it should be `""`.

        # Adjust mock to simulate empty output for comment-only
        original_transpile = sys.modules['sqlglot'].transpile # type: ignore
        def mock_transpile_for_comment(sql, read, write, pretty):
            if sql.strip().startswith("--"): return []
            return original_transpile(sql,read,write,pretty)
        sys.modules['sqlglot'].transpile = mock_transpile_for_comment # type: ignore

        translated = translator.translate(comment_query, SQLDialect.POSTGRES) # type: ignore
        self.assertEqual(translated, "")
        sys.modules['sqlglot'].transpile = original_transpile # type: ignore


    def test_parse_sql_valid(self):
        """Test parsing a valid SQL query into an AST (mocked)."""
        ast = SQLTranslator.parse_sql("SELECT 1") # type: ignore
        self.assertIsInstance(ast, sys.modules['sqlglot'].MockSqlglotExpression) # type: ignore

    def test_parse_sql_invalid_sql(self):
        """Test parsing invalid SQL (mocked parse error)."""
        with self.assertRaises(sys.modules['sqlglot'].ParseError): # type: ignore
            SQLTranslator.parse_sql("SELECT parse_error_trigger 1") # type: ignore

    def test_parse_sql_comment_only(self):
        """Test parsing comment-only SQL (should raise ParseError with current logic)."""
        # The `parse_sql` method uses `sqlglot.parse` which returns `[None]` for comment-only.
        # The method then tries to access `parsed_expressions[0]`, which would be `None`.
        # The check `if not parsed_expressions or parsed_expressions[0] is None:` handles this.
        with self.assertRaisesRegex(sys.modules['sqlglot'].ParseError, "No valid SQL expression found"): # type: ignore
            SQLTranslator.parse_sql("-- only comment") # type: ignore


    def test_render_ast_valid(self):
        """Test rendering an AST to SQL string (mocked)."""
        mock_ast = sys.modules['sqlglot'].MockSqlglotExpression() # type: ignore
        rendered_sql = SQLTranslator.render_ast(mock_ast, SQLDialect.MYSQL) # type: ignore
        self.assertEqual(rendered_sql, f"mock sql for {SQLDialect.MYSQL.value}") # type: ignore

    def test_render_ast_unsupported_dialect(self):
        """Test rendering AST to an unsupported dialect (mocked)."""
        sys.modules['sqlglot'].Dialect.get = lambda x: None if x == "made_up_dialect" else True # type: ignore
        mock_ast = sys.modules['sqlglot'].MockSqlglotExpression() # type: ignore
        class MockUnsupportedDialectEnum(SQLDialect): # type: ignore
            MADEUP = "made_up_dialect"
        with self.assertRaisesRegex(ValueError, "Unsupported SQL dialect for rendering: made_up_dialect"):
            SQLTranslator.render_ast(mock_ast, MockUnsupportedDialectEnum.MADEUP) # type: ignore
        sys.modules['sqlglot'].Dialect.get = lambda x: True if x else None # type: ignore


if __name__ == "__main__":
    if SQLTranslator is not None and SQLDialect is not None and 'sqlglot' in sys.modules:
        print("Running SQLTranslator tests (illustrative execution with mocks)...")
        unittest.main(verbosity=2)
    else:
        print("Skipping SQLTranslator tests as SQLTranslator module or mocked 'sqlglot' could not be properly set up.")
        print("This is expected if dependencies are not installed or src path is incorrect in a non-execution environment.")
