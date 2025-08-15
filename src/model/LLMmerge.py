"""
LLM SQL Query Merger

This module implements an advanced LLM system that takes a single prompt containing
multiple SQL queries and merges them into a single, unified SQL query through 
intelligent concatenation and combination strategies.

Key Features:
- Detects and extracts multiple SQL queries from text
- Merges queries using different strategies (UNION, JOIN, subqueries)
- Uses phi-4-mini-instruct for intelligent SQL understanding
- Supports complex query combination patterns
- Validates merged queries using SQLGlot
"""

import json
import re
import torch
from typing import Dict, List, Optional, Union, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline
)
import logging

# Import for SQL validation
try:
    import sqlglot
    from sqlglot import parse_one, transpile
    from sqlglot.errors import ParseError
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    logging.warning("SQLGlot non disponibile. Validazione SQL disabilitata.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLQueryExtractor:
    """
    Utility class for extracting SQL queries from text using regex patterns.
    """
    
    def __init__(self):
        # Common SQL keywords that typically start queries
        self.sql_keywords = [
            r'\bSELECT\b', r'\bINSERT\b', r'\bUPDATE\b', r'\bDELETE\b',
            r'\bCREATE\b', r'\bDROP\b', r'\bALTER\b', r'\bWITH\b'
        ]
        
        # Patterns to identify SQL queries
        self.sql_patterns = [
            # Pattern for queries ending with semicolon
            r'((?:' + '|'.join(self.sql_keywords) + r').*?;)',
            # Pattern for queries without semicolon but with typical SQL structure
            r'((?:' + '|'.join(self.sql_keywords) + r').*?(?=\n\s*(?:' + '|'.join(self.sql_keywords) + r')|$))'
        ]
    
    def extract_sql_queries(self, text: str) -> List[str]:
        """
        Extract all SQL queries from the input text.
        
        Args:
            text: Input text containing multiple SQL queries
            
        Returns:
            List of extracted SQL query strings
        """
        queries = []
        
        # Clean the text
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Remove block comments
        text = re.sub(r'--.*?$', '', text, flags=re.MULTILINE)  # Remove line comments
        
        # Extract queries using patterns
        for pattern in self.sql_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                cleaned_query = self._clean_query(match)
                if cleaned_query and len(cleaned_query) > 10:  # Minimum length filter
                    queries.append(cleaned_query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            query_normalized = re.sub(r'\s+', ' ', query.strip().upper())
            if query_normalized not in seen:
                seen.add(query_normalized)
                unique_queries.append(query.strip())
        
        return unique_queries
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize a SQL query."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove trailing semicolon for processing
        query = query.rstrip(';').strip()
        
        return query


class LLMSQLMerger:
    """
    LLM-powered SQL query merger that intelligently combines multiple SQL queries
    into a single unified query using different merging strategies.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM SQL Merger.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference ("cpu", "cuda", etc.). Auto-detected if None.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.extractor = SQLQueryExtractor()
        self._load_model()
        
        # System prompt for SQL merging
        self.system_prompt = (
            "You are an expert SQL developer. Your task is to analyze multiple SQL queries "
            "and merge them into a single, unified SQL query. Use appropriate techniques like "
            "UNION, JOIN, subqueries, or CTEs to combine the queries logically. "
            "Always return valid SQL syntax."
        )
    
    def _load_model(self):
        """Load the model and create the pipeline."""
        try:
            logger.info(f"Loading LLM SQL Merger model: {self.model_id}")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                return_full_text=False,
                max_new_tokens=1000,  # Allow for longer SQL queries
                do_sample=True,
                temperature=0.3,      # Low temperature for more deterministic output
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            logger.info("LLM SQL Merger model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM SQL Merger model: {str(e)}")
            raise
    
    def merge_sql_queries(self, input_text: str) -> Dict[str, Union[str, List[str], bool]]:
        """
        Extract and merge multiple SQL queries from input text.
        
        Args:
            input_text: Text containing multiple SQL queries
            
        Returns:
            Dictionary with merged query, extracted queries, and validation info
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded properly")
        
        try:
            # Extract individual SQL queries
            extracted_queries = self.extractor.extract_sql_queries(input_text)
            
            if not extracted_queries:
                return {
                    "success": False,
                    "error": "No SQL queries found in the input text",
                    "extracted_queries": [],
                    "merged_query": "",
                    "is_valid": False
                }
            
            if len(extracted_queries) == 1:
                # Only one query found, return as-is
                single_query = extracted_queries[0]
                validation = self._validate_sql(single_query) if SQL_VALIDATION_AVAILABLE else {"valid": None}
                
                return {
                    "success": True,
                    "extracted_queries": extracted_queries,
                    "merged_query": single_query,
                    "is_valid": validation.get("valid", None),
                    "validation_error": validation.get("error"),
                    "merge_strategy": "single_query"
                }
            
            # Multiple queries found, merge them
            logger.info(f"Found {len(extracted_queries)} SQL queries to merge")
            
            # Create prompt for merging
            prompt = self._create_merge_prompt(extracted_queries)
            
            # Generate merged query
            result = self.pipeline(prompt)
            merged_sql = self._extract_sql_from_response(result[0]["generated_text"])
            
            # Validate the merged query
            validation = self._validate_sql(merged_sql) if SQL_VALIDATION_AVAILABLE else {"valid": None}
            
            # Determine merge strategy used
            merge_strategy = self._detect_merge_strategy(merged_sql)
            
            response = {
                "success": True,
                "extracted_queries": extracted_queries,
                "merged_query": merged_sql,
                "is_valid": validation.get("valid", None),
                "validation_error": validation.get("error"),
                "merge_strategy": merge_strategy,
                "prompt_used": prompt
            }
            
            logger.info(f"Successfully merged {len(extracted_queries)} queries using {merge_strategy}")
            return response
            
        except Exception as e:
            logger.error(f"Error during SQL merging: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_queries": [],
                "merged_query": "",
                "is_valid": False
            }
    
    def _create_merge_prompt(self, queries: List[str]) -> str:
        """
        Create a prompt for the LLM to merge the SQL queries.
        
        Args:
            queries: List of SQL queries to merge
            
        Returns:
            Formatted prompt for the LLM
        """
        queries_text = ""
        for i, query in enumerate(queries, 1):
            queries_text += f"\nQuery {i}:\n{query}\n"
        
        prompt = f"""System: {self.system_prompt}

I have the following {len(queries)} SQL queries that need to be merged into a single unified query:
{queries_text}

Please analyze these queries and combine them into one optimized SQL query. Consider:
1. If they select from similar tables, use UNION or UNION ALL
2. If they involve related data, use JOIN operations  
3. If they have dependencies, use subqueries or CTEs
4. Ensure the final query is syntactically correct and logically sound

Merged SQL Query:"""

        return prompt
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract clean SQL query from the LLM response.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Clean SQL query string
        """
        # Remove common prefixes/suffixes
        response = response.strip()
        
        # Try to find SQL query in the response
        sql_patterns = [
            r'```sql\s*(.*?)\s*```',  # Markdown SQL blocks
            r'```\s*(.*?)\s*```',     # Generic code blocks
            r'((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE).*?)(?:\n\n|$)'  # Direct SQL
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(1).strip()
                # Clean up the SQL
                sql = re.sub(r'\s+', ' ', sql)
                return sql
        
        # If no pattern matches, try to clean the entire response
        cleaned = re.sub(r'^.*?(?=SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)', '', response, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        
        return cleaned
    
    def _detect_merge_strategy(self, merged_query: str) -> str:
        """
        Detect which merge strategy was used in the merged query.
        
        Args:
            merged_query: The merged SQL query
            
        Returns:
            Strategy name
        """
        query_upper = merged_query.upper()
        
        if 'UNION' in query_upper:
            return "UNION"
        elif 'JOIN' in query_upper:
            return "JOIN"
        elif 'WITH' in query_upper:
            return "CTE"
        elif re.search(r'\(\s*SELECT.*SELECT.*\)', query_upper, re.DOTALL):
            return "SUBQUERY"
        else:
            return "CONCATENATION"
    
    def _validate_sql(self, sql_query: str) -> Dict[str, Union[bool, str]]:
        """
        Validate SQL query syntax using SQLGlot.
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Dictionary with validation results
        """
        if not SQL_VALIDATION_AVAILABLE:
            return {"valid": None, "error": "SQLGlot not available"}
        
        try:
            # Try to parse the query
            parsed = parse_one(sql_query, dialect="sql")
            return {"valid": True, "error": None, "parsed": str(parsed)}
        except ParseError as e:
            return {"valid": False, "error": f"Parse error: {str(e)}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def batch_merge(self, input_texts: List[str]) -> List[Dict[str, Union[str, List[str], bool]]]:
        """
        Merge SQL queries from multiple input texts in batch.
        
        Args:
            input_texts: List of texts containing SQL queries
            
        Returns:
            List of merge results
        """
        results = []
        for i, text in enumerate(input_texts):
            logger.info(f"Processing batch item {i+1}/{len(input_texts)}")
            result = self.merge_sql_queries(text)
            results.append(result)
        return results


class LLMSQLMergerSystem:
    """
    High-level interface for the LLM SQL Merger system.
    """
    
    def __init__(self, model_id: str = "microsoft/phi-4-mini-instruct", device: Optional[str] = None):
        """
        Initialize the LLM SQL Merger System.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to use for inference
        """
        logger.info("Initializing LLM SQL Merger System")
        self.merger = LLMSQLMerger(model_id, device)
        logger.info("LLM SQL Merger System initialized successfully")
    
    def merge_queries(self, input_text: str, detailed: bool = False) -> Union[str, Dict]:
        """
        Merge SQL queries from input text.
        
        Args:
            input_text: Text containing multiple SQL queries
            detailed: If True, return detailed analysis; if False, return just merged query
            
        Returns:
            Merged SQL query string or detailed analysis dictionary
        """
        result = self.merger.merge_sql_queries(input_text)
        
        if detailed:
            return result
        else:
            return result.get("merged_query", "") if result.get("success") else ""
    
    def batch_merge(self, input_texts: List[str], detailed: bool = False) -> List[Union[str, Dict]]:
        """
        Merge SQL queries from multiple input texts.
        
        Args:
            input_texts: List of texts containing SQL queries
            detailed: If True, return detailed analysis; if False, return just merged queries
            
        Returns:
            List of merged queries or detailed analyses
        """
        results = self.merger.batch_merge(input_texts)
        
        if detailed:
            return results
        else:
            return [r.get("merged_query", "") if r.get("success") else "" for r in results]


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM SQL Query Merger")
    parser.add_argument("--input", type=str, help="Input text containing multiple SQL queries")
    parser.add_argument("--file", type=str, help="Input file containing SQL queries")
    parser.add_argument("--model", type=str, default="microsoft/phi-4-mini-instruct",
                       help="Model ID to use")
    parser.add_argument("--device", type=str, help="Device to use (cpu/cuda)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis including validation")
    parser.add_argument("--batch", type=str,
                       help="Path to file with multiple inputs for batch processing")
    
    args = parser.parse_args()
    
    # Initialize system
    system = LLMSQLMergerSystem(args.model, args.device)
    
    if args.batch:
        # Batch processing mode
        try:
            with open(args.batch, 'r', encoding='utf-8') as f:
                batch_inputs = [line.strip() for line in f if line.strip()]
            
            results = system.batch_merge(batch_inputs, detailed=args.detailed)
            
            print("\n=== BATCH SQL MERGE RESULTS ===")
            for i, result in enumerate(results):
                print(f"\n--- Input {i+1} ---")
                if args.detailed:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(f"Merged Query: {result}")
                    
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
    
    elif args.file:
        # File input mode
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
            
            print(f"\n=== PROCESSING FILE: {args.file} ===")
            result = system.merge_queries(input_text, detailed=True)
            
            if result["success"]:
                print(f"\nExtracted {len(result['extracted_queries'])} queries:")
                for i, query in enumerate(result['extracted_queries'], 1):
                    print(f"\n{i}. {query}")
                
                print(f"\n= MERGED QUERY ({result['merge_strategy']}):")
                print("=" * 60)
                print(result['merged_query'])
                print("=" * 60)
                
                if result.get('is_valid') is not None:
                    status = " Valid" if result['is_valid'] else "L Invalid"
                    print(f"\nValidation: {status}")
                    if result.get('validation_error'):
                        print(f"Error: {result['validation_error']}")
            else:
                print(f"L Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
    
    elif args.input:
        # Single input mode
        print(f"\n=== MERGING SQL QUERIES ===")
        print(f"Input: {args.input}")
        print("=" * 50)
        
        result = system.merge_queries(args.input, detailed=True)
        
        if result["success"]:
            print(f"\n== ANALYSIS:")
            print(f"Extracted queries: {len(result['extracted_queries'])}")
            print(f"Merge strategy: {result['merge_strategy']}")
            
            print(f"\n== EXTRACTED QUERIES:")
            for i, query in enumerate(result['extracted_queries'], 1):
                print(f"{i}. {query}")
            
            print(f"\n== MERGED QUERY:")
            print("=" * 50)
            print(result['merged_query'])
            print("=" * 50)
            
            if result.get('is_valid') is not None:
                status = " Valid" if result['is_valid'] else "L Invalid" 
                print(f"\nValidation: {status}")
                if result.get('validation_error'):
                    print(f"Error: {result['validation_error']}")
        else:
            print(f"L Error: {result['error']}")
    
    else:
        # Demo mode
        print("\n=== DEMO MODE ===")
        print("Running demonstrations with example SQL queries...\n")
        
        demo_examples = [
            # Example 1: Multiple SELECT queries
            """
            Get all users from the database:
            SELECT * FROM users WHERE age > 25;
            
            Also get all active products:
            SELECT name, price FROM products WHERE status = 'active';
            """,
            
            # Example 2: Related queries that can be JOINed
            """
            SELECT user_id, name FROM users;
            SELECT user_id, order_total FROM orders;
            """,
            
            # Example 3: Complex queries with different operations
            """
            CREATE TABLE temp_sales AS SELECT * FROM sales WHERE date > '2024-01-01';
            SELECT category, SUM(amount) FROM temp_sales GROUP BY category;
            DROP TABLE temp_sales;
            """
        ]
        
        for i, example in enumerate(demo_examples, 1):
            print(f"\n--- Example {i} ---")
            print(f"Input text: {example.strip()}")
            
            result = system.merge_queries(example, detailed=True)
            
            if result["success"]:
                print(f"Strategy: {result['merge_strategy']}")
                print(f"Queries found: {len(result['extracted_queries'])}")
                print(f"Merged query: {result['merged_query']}")
                validation = "" if result.get('is_valid') else "L" if result.get('is_valid') is False else "?"
                print(f"Valid: {validation}")
            else:
                print(f"Error: {result['error']}")