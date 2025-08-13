"""
Module for translating SQL queries between dialects using sqlglot.

Optimized version with enhanced error handling, better performance, and 
comprehensive SQLGlot API usage.

See: https://github.com/tobymao/sqlglot
"""

import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Union

__all__ = ["SQLDialect", "SQLTranslator", "TranslationResult"]

try:
    import sqlglot
    from sqlglot import Dialect, ParseError, transpile, parse_one, parse
    from sqlglot.errors import SqlglotError, TokenError, UnsupportedError
    import sqlglot.expressions as exp
except ImportError:
    raise ImportError(
        "sqlglot library is not installed. Please run: pip install sqlglot"
    )

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TranslationResult:
    """Encapsulates translation results with metadata."""
    
    def __init__(
        self, 
        translated_sql: str, 
        source_dialect: Optional[str], 
        target_dialect: str,
        original_sql: str,
        warnings: Optional[List[str]] = None
    ):
        self.translated_sql = translated_sql
        self.source_dialect = source_dialect
        self.target_dialect = target_dialect
        self.original_sql = original_sql
        self.warnings = warnings or []
    
    def __str__(self) -> str:
        return self.translated_sql
    
    def __repr__(self) -> str:
        return f"TranslationResult(source={self.source_dialect}, target={self.target_dialect})"


class SQLDialect(Enum):
    """Enum of supported SQL dialects mapped to sqlglot dialect strings."""
    BIGQUERY = "bigquery"
    CLICKHOUSE = "clickhouse" 
    DATABRICKS = "databricks"
    DREMIO = "dremio"
    DRILL = "drill"
    DUCKDB = "duckdb"
    HIVE = "hive"
    MATERIALIZE = "materialize"
    MYSQL = "mysql"
    ORACLE = "oracle"
    POSTGRES = "postgres"
    PRESTO = "presto"
    PRQL = "prql"
    REDSHIFT = "redshift"
    RISINGWAVE = "risingwave"
    SNOWFLAKE = "snowflake"
    SPARK = "spark"
    SPARK2 = "spark2"
    SQLITE = "sqlite"
    STARROCKS = "starrocks"
    TABLEAU = "tableau"
    TERADATA = "teradata"
    TRINO = "trino"
    TSQL = "tsql"

    @classmethod
    def from_string(cls, s: str) -> 'SQLDialect':
        """Create SQLDialect from string with error handling."""
        try:
            return cls[s.upper()]
        except KeyError:
            available = [member.name.lower() for member in cls]
            raise ValueError(
                f"Unsupported SQL dialect: '{s}'. "
                f"Supported dialects: {', '.join(sorted(available))}"
            )

    def to_sqlglot_dialect(self) -> str:
        """Convert to sqlglot dialect string."""
        return self.value
    
    @classmethod
    def get_supported_dialects(cls) -> List[str]:
        """Get list of all supported dialect names."""
        return [member.name.lower() for member in cls]
    
    def is_valid(self) -> bool:
        """Check if this dialect is supported by current SQLGlot version."""
        try:
            return Dialect.get_or_raise(self.value) is not None
        except Exception:
            return False


class SQLTranslator:
    """
    Optimized SQL translator using SQLGlot with enhanced error handling,
    better performance, and comprehensive API coverage.
    """

    def __init__(
        self, 
        source_dialect: Optional[SQLDialect] = None,
        validate_dialects: bool = True
    ):
        """
        Initialize SQLTranslator.
        
        Args:
            source_dialect: Default source dialect for translations
            validate_dialects: Whether to validate dialect support on init
        """
        self.source_dialect = source_dialect
        self.source_dialect_str = source_dialect.to_sqlglot_dialect() if source_dialect else None
        self._warnings: List[str] = []

        if validate_dialects and source_dialect and not source_dialect.is_valid():
            raise ValueError(f"Dialect '{source_dialect.value}' is not supported by current SQLGlot version")

        logger.info(f"SQLTranslator initialized. Source dialect: {self.source_dialect_str or 'auto-detect'}")

    def translate(
        self,
        sql_query: str,
        target_dialect: SQLDialect,
        pretty: bool = True,
        source_dialect_override: Optional[SQLDialect] = None,
        **transpile_options
    ) -> TranslationResult:
        """
        Translate SQL query with comprehensive error handling and metadata.
        
        Args:
            sql_query: The SQL query to translate
            target_dialect: Target dialect for translation
            pretty: Whether to format output SQL nicely
            source_dialect_override: Override default source dialect
            **transpile_options: Additional options passed to sqlglot.transpile
            
        Returns:
            TranslationResult with translated SQL and metadata
        """
        if not sql_query.strip():
            raise ValueError("SQL query cannot be empty.")

        # Clear previous warnings
        self._warnings.clear()

        # Determine dialects
        source_dialect_str = (
            source_dialect_override.to_sqlglot_dialect() 
            if source_dialect_override 
            else self.source_dialect_str
        )
        target_dialect_str = target_dialect.to_sqlglot_dialect()

        # Validate target dialect
        if not target_dialect.is_valid():
            raise ValueError(f"Target dialect '{target_dialect_str}' is not supported")

        # Validate source dialect if specified
        if source_dialect_override and not source_dialect_override.is_valid():
            raise ValueError(f"Source dialect '{source_dialect_str}' is not supported")

        try:
            # Use sqlglot.transpile with enhanced options
            result = transpile(
                sql_query,
                read=source_dialect_str,
                write=target_dialect_str,
                pretty=pretty,
                **transpile_options
            )
            
            if not result:
                raise ValueError("Translation produced no results")
                
            # Handle multiple statements more carefully
            if len(result) > 1:
                warning = f"Multiple SQL statements detected ({len(result)}), joining with semicolons"
                self._warnings.append(warning)
                logger.warning(warning)
                translated_sql = ";\n".join(result)
            else:
                translated_sql = result[0]
            
            return TranslationResult(
                translated_sql=translated_sql,
                source_dialect=source_dialect_str,
                target_dialect=target_dialect_str,
                original_sql=sql_query,
                warnings=self._warnings.copy()
            )

        except ParseError as e:
            logger.error(f"SQL parsing error: {e}")
            raise ParseError(f"Failed to parse SQL: {str(e)}") from e
        except TokenError as e:
            logger.error(f"SQL tokenization error: {e}")
            raise TokenError(f"Invalid SQL syntax: {str(e)}") from e
        except UnsupportedError as e:
            logger.error(f"Unsupported SQL feature: {e}")
            raise UnsupportedError(f"SQL feature not supported in target dialect: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            raise RuntimeError(f"Translation failed: {str(e)}") from e

    def translate_simple(
        self,
        sql_query: str,
        target_dialect: SQLDialect,
        **kwargs
    ) -> str:
        """
        Simplified translation method returning only the SQL string.
        
        Args:
            sql_query: The SQL query to translate
            target_dialect: Target dialect for translation
            **kwargs: Additional options passed to translate()
            
        Returns:
            Translated SQL string
        """
        result = self.translate(sql_query, target_dialect, **kwargs)
        return result.translated_sql

    @staticmethod
    def parse_sql_single(
        sql_query: str, 
        dialect: Optional[SQLDialect] = None,
        **parse_options
    ) -> exp.Expression:
        """
        Parse a single SQL statement using parse_one (optimized).
        
        Args:
            sql_query: SQL query to parse
            dialect: Dialect for parsing
            **parse_options: Additional parsing options
            
        Returns:
            Parsed expression tree
        """
        dialect_str = dialect.to_sqlglot_dialect() if dialect else None
        
        try:
            # Use parse_one for better performance with single statements
            parsed = parse_one(sql_query, read=dialect_str, **parse_options)
            
            if parsed is None:
                raise ParseError("No valid SQL expression found")
                
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse SQL: {e}")
            raise

    @staticmethod
    def parse_sql_multiple(
        sql_query: str, 
        dialect: Optional[SQLDialect] = None,
        **parse_options
    ) -> List[Optional[exp.Expression]]:
        """
        Parse multiple SQL statements.
        
        Args:
            sql_query: SQL query to parse (may contain multiple statements)
            dialect: Dialect for parsing
            **parse_options: Additional parsing options
            
        Returns:
            List of parsed expression trees
        """
        dialect_str = dialect.to_sqlglot_dialect() if dialect else None
        
        try:
            parsed = parse(sql_query, read=dialect_str, **parse_options)
            
            if not parsed:
                raise ParseError("No valid SQL expressions found")
                
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse SQL: {e}")
            raise

    @staticmethod
    def render_ast(
        ast_expression: exp.Expression, 
        dialect: SQLDialect, 
        pretty: bool = True,
        **sql_options
    ) -> str:
        """
        Render AST to SQL string with enhanced options.
        
        Args:
            ast_expression: The AST expression to render
            dialect: Target dialect for rendering
            pretty: Whether to format nicely
            **sql_options: Additional SQL generation options
            
        Returns:
            Generated SQL string
        """
        if not dialect.is_valid():
            raise ValueError(f"Unsupported rendering dialect: {dialect.value}")
            
        try:
            return ast_expression.sql(
                dialect=dialect.to_sqlglot_dialect(), 
                pretty=pretty,
                **sql_options
            )
        except Exception as e:
            logger.error(f"Failed to render AST: {e}")
            raise

    def batch_translate(
        self,
        sql_queries: List[str],
        target_dialect: SQLDialect,
        **translate_options
    ) -> List[TranslationResult]:
        """
        Translate multiple SQL queries in batch.
        
        Args:
            sql_queries: List of SQL queries to translate
            target_dialect: Target dialect for all translations
            **translate_options: Options passed to translate()
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        
        for i, query in enumerate(sql_queries):
            try:
                result = self.translate(query, target_dialect, **translate_options)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to translate query {i+1}: {e}")
                # Create error result
                error_result = TranslationResult(
                    translated_sql=f"-- Translation failed: {str(e)}",
                    source_dialect=self.source_dialect_str,
                    target_dialect=target_dialect.to_sqlglot_dialect(),
                    original_sql=query,
                    warnings=[f"Translation error: {str(e)}"]
                )
                results.append(error_result)
                
        return results

    def get_warnings(self) -> List[str]:
        """Get warnings from the last translation operation."""
        return self._warnings.copy()

    @staticmethod
    def validate_sql(
        sql_query: str, 
        dialect: Optional[SQLDialect] = None
    ) -> Dict[str, Any]:
        """
        Validate SQL syntax and return detailed information.
        
        Args:
            sql_query: SQL query to validate
            dialect: Dialect for validation
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "statement_count": 0,
            "statement_types": []
        }
        
        try:
            parsed = SQLTranslator.parse_sql_multiple(sql_query, dialect)
            result["is_valid"] = True
            result["statement_count"] = len([p for p in parsed if p is not None])
            result["statement_types"] = [type(p).__name__ for p in parsed if p is not None]
            
        except Exception as e:
            result["errors"].append(str(e))
            
        return result
    


if __name__ == "__main__":
    
    def demonstrate_optimized_features():
        """Demonstrate the optimized SQLTranslator features."""
        print("=== SQLTranslator Optimized Demo ===\n")
        
        # Initialize translator
        translator = SQLTranslator(source_dialect=SQLDialect.SQLITE)
        
        # Test queries
        queries = [
            "SELECT id, name, email FROM users WHERE active = 1",
            "SELECT DATE('now'), COUNT(*) FROM orders GROUP BY DATE(created_at)",
            "WITH cte AS (SELECT * FROM products) SELECT * FROM cte",
            "INVALID SQL QUERY"  # Test error handling
        ]
        
        print("1. Basic Translation with TranslationResult:")
        for query in queries[:3]:
            try:
                result = translator.translate(query, SQLDialect.POSTGRES)
                print(f"Original: {query}")
                print(f"Translated: {result.translated_sql}")
                print(f"Source: {result.source_dialect}, Target: {result.target_dialect}")
                if result.warnings:
                    print(f"Warnings: {result.warnings}")
                print("-" * 50)
            except Exception as e:
                print(f"Translation failed for '{query}': {e}")
        
        print("\n2. Batch Translation:")
        batch_results = translator.batch_translate(queries, SQLDialect.MYSQL)
        for i, result in enumerate(batch_results):
            print(f"Query {i+1}: {result.translated_sql[:50]}...")
            if result.warnings:
                print(f"  Warnings: {result.warnings}")
        
        print("\n3. SQL Validation:")
        for query in queries:
            validation = SQLTranslator.validate_sql(query, SQLDialect.SQLITE)
            print(f"Query: {query[:30]}...")
            print(f"  Valid: {validation['is_valid']}")
            print(f"  Statements: {validation['statement_count']}")
            if validation['errors']:
                print(f"  Errors: {validation['errors']}")
        
        print("\n4. Enhanced AST Operations:")
        try:
            # Single statement parsing
            ast = SQLTranslator.parse_sql_single(
                "SELECT column1, column2 FROM table1 WHERE id > 100"
            )
            print(f"AST type: {type(ast).__name__}")
            
            # Render to different dialects
            for dialect in [SQLDialect.BIGQUERY, SQLDialect.SNOWFLAKE, SQLDialect.SPARK]:
                rendered = SQLTranslator.render_ast(ast, dialect, pretty=True)
                print(f"{dialect.name}: {rendered}")
                
        except Exception as e:
            print(f"AST operations failed: {e}")
        
        print("\n5. Dialect Information:")
        print(f"Supported dialects: {', '.join(SQLDialect.get_supported_dialects())}")
        
        # Test dialect validation
        for dialect in [SQLDialect.POSTGRES, SQLDialect.MYSQL, SQLDialect.BIGQUERY]:
            print(f"{dialect.name} is valid: {dialect.is_valid()}")

    def translate_to_all_dialects(sqlite_query: str) -> None:
        """
        Enhanced version with better error handling and metadata reporting.
        """
        print(f"\n=== Translating SQLite query to all supported dialects ===")
        print(f"Original query: {sqlite_query}\n")
        
        translator = SQLTranslator(source_dialect=SQLDialect.SQLITE)
        successful_translations = 0
        failed_translations = 0

        for target_dialect in SQLDialect:
            if target_dialect == SQLDialect.SQLITE:
                continue  # Skip self-translation
                
            print(f"--- {target_dialect.name} ---")
            try:
                # Use the new translate method
                result = translator.translate(sqlite_query, target_dialect=target_dialect)
                print(f"✅ SUCCESS: {result.translated_sql}")
                
                if result.warnings:
                    print(f"⚠️  Warnings: {', '.join(result.warnings)}")
                
                successful_translations += 1
                
                # Test round-trip translation
                reverse_translator = SQLTranslator(source_dialect=target_dialect)
                try:
                    reverse_result = reverse_translator.translate(
                        result.translated_sql, 
                        SQLDialect.SQLITE
                    )
                    print(f"🔄 Round-trip: {reverse_result.translated_sql}")
                except Exception:
                    print("🔄 Round-trip: Failed")
                    
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                failed_translations += 1
            
            print()

        print(f"Summary: {successful_translations} successful, {failed_translations} failed")

    # Run demonstrations
    demonstrate_optimized_features()
    
    print("\n" + "="*60)
    
    # Test round-trip translations
    test_query = "SELECT id, name, COUNT(*) as count FROM users WHERE active = 1 GROUP BY id, name HAVING count > 5"
    translate_to_all_dialects(test_query)
    
    print("\n=== Optimized SQLTranslator Demo Complete ===")
