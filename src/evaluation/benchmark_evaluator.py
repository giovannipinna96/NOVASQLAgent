"""
Benchmark Evaluation Module for SQL Agents.
This module provides functionalities to evaluate generated SQL queries against gold queries,
including execution accuracy if results are available or can be simulated.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

# May use the SQLExecutor to run queries if actual execution is part of evaluation.
# from ..execution.sql_executor import SQLExecutor, SQLExecutionResult
# For comparing SQL structure, a library like sqlglot could be useful.
# import sqlglot

logger = logging.getLogger(__name__)

class EvaluationResult:
    """Represents the evaluation outcome for a single query pair."""
    def __init__(self,
                 question_id: Any,
                 predicted_sql: str,
                 gold_sql: str,
                 exact_match: bool,
                 execution_match: Optional[bool] = None, # If execution results are compared
                 syntax_valid: Optional[bool] = None, # If syntax check is performed
                 predicted_results: Optional[List[Dict]] = None,
                 gold_results: Optional[List[Dict]] = None,
                 error_message: Optional[str] = None):
        self.question_id = question_id
        self.predicted_sql = predicted_sql
        self.gold_sql = gold_sql
        self.exact_match = exact_match # String exact match of SQL
        self.execution_match = execution_match # Do execution results match?
        self.syntax_valid = syntax_valid # Is predicted SQL syntactically valid?
        self.predicted_results = predicted_results
        self.gold_results = gold_results
        self.error_message = error_message

    def __repr__(self) -> str:
        return (f"<EvaluationResult id={self.question_id}, exact_match={self.exact_match}, "
                f"exec_match={self.execution_match}, syntax_valid={self.syntax_valid}>")

class BenchmarkEvaluator:
    """
    Evaluates agent-generated SQL queries against a benchmark (e.g., Spider).
    """
    def __init__(self, sql_executor: Optional[Any] = None, db_name: Optional[str] = None):
        """
        Initializes the BenchmarkEvaluator.

        Args:
            sql_executor: An optional SQLExecutor instance for running queries.
                          If None, execution-based evaluation will be skipped.
            db_name: Identifier for the database against which queries are run (for context).
        """
        self.sql_executor = sql_executor
        self.db_name = db_name
        logger.info(f"BenchmarkEvaluator initialized. SQL Executor: {'Provided' if sql_executor else 'Not Provided'}")

    def compare_sql_queries(self, predicted_sql: str, gold_sql: str) -> bool:
        """
        Compares two SQL queries for exact match (case-insensitive, ignoring minor whitespace).
        A more advanced version would use SQL parsing and canonicalization (e.g., with sqlglot).
        """
        # Simple normalization: lowercase, strip whitespace, replace multiple spaces with one
        norm_pred = ' '.join(predicted_sql.lower().strip().split())
        norm_gold = ' '.join(gold_sql.lower().strip().split())

        # Remove trailing semicolons for comparison
        if norm_pred.endswith(';'): norm_pred = norm_pred[:-1]
        if norm_gold.endswith(';'): norm_gold = norm_gold[:-1]

        return norm_pred == norm_gold

    def compare_execution_results(self, pred_results: List[Dict], gold_results: List[Dict]) -> bool:
        """
        Compares two sets of query execution results.
        Order of rows and columns might matter depending on benchmark rules.
        This is a simplified comparison (set of tuples of sorted items).
        """
        if len(pred_results) != len(gold_results):
            return False
        if not pred_results: # Both empty
            return True

        # Convert rows to a comparable format (e.g., tuple of sorted (key, value) pairs)
        # This makes row comparison order-insensitive for columns, and then set comparison makes it order-insensitive for rows.
        def _normalize_row(row: Dict) -> Tuple:
            return tuple(sorted(row.items()))

        norm_pred_set = set(map(_normalize_row, pred_results))
        norm_gold_set = set(map(_normalize_row, gold_results))

        return norm_pred_set == norm_gold_set

    def check_sql_syntax(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Checks SQL syntax. Conceptually, this might use a parser or try to execute.
        Using SQLExecutor's execution attempt can serve as a proxy for syntax check.
        """
        if not self.sql_executor:
            logger.warning("No SQL executor provided, cannot perform syntax check via execution.")
            return False, "SQL Executor not available for syntax check." # Or True, None if we don't want to penalize

        # A common way to check syntax is to try to "EXPLAIN" the query or execute with LIMIT 0.
        # For simplicity, we can try a dry run or a minimal execution.
        # The actual SQLSandbox tries to execute; if it fails with a syntax error, that's our check.

        # Conceptual: Some databases allow "EXPLAIN" or "PARSE" type commands.
        # For now, we'll rely on the fact that execute_sql will return an error for syntax issues.
        # This is a placeholder; a real syntax check might use a parser.
        # Let's assume if sql_executor.execute_sql fails with a syntax-like error, it's invalid.
        # This method itself doesn't call execute_sql; it's more of a note for evaluate_one.
        logger.info(f"Conceptual syntax check for: {sql_query[:100]}... (Relies on execution attempt)")
        return True, None # Assume valid unless execution proves otherwise

    def evaluate_one(self, question_id: Any, predicted_sql: str, gold_sql: str,
                     gold_results_data: Optional[List[Dict]] = None) -> EvaluationResult:
        """
        Evaluates a single predicted SQL query against a gold SQL query.

        Args:
            question_id: An identifier for the question/query pair.
            predicted_sql: The SQL query generated by the agent.
            gold_sql: The ground truth SQL query.
            gold_results_data: Optional pre-fetched execution results for the gold query.

        Returns:
            An EvaluationResult object.
        """
        logger.info(f"Evaluating QID {question_id}: Predicted: '{predicted_sql[:100]}...' Gold: '{gold_sql[:100]}...'")

        exact_match = self.compare_sql_queries(predicted_sql, gold_sql)

        syntax_valid: Optional[bool] = None
        execution_match: Optional[bool] = None
        pred_exec_results: Optional[List[Dict]] = None
        eval_error_message: Optional[str] = None

        if self.sql_executor:
            # Attempt to execute predicted SQL
            logger.debug(f"Executing predicted SQL for QID {question_id}...")
            pred_exec = self.sql_executor.execute_sql(predicted_sql)

            if pred_exec.success:
                syntax_valid = True # If it ran, syntax is valid
                pred_exec_results = pred_exec.results

                # Now execute gold SQL (or use provided gold_results_data)
                current_gold_results = gold_results_data
                if current_gold_results is None:
                    logger.debug(f"Executing gold SQL for QID {question_id}...")
                    gold_exec = self.sql_executor.execute_sql(gold_sql)
                    if gold_exec.success:
                        current_gold_results = gold_exec.results
                    else:
                        eval_error_message = f"Gold SQL execution failed: {gold_exec.error_message}"
                        logger.error(eval_error_message)
                        # If gold SQL fails, can't determine execution match.

                if current_gold_results is not None and pred_exec_results is not None:
                    execution_match = self.compare_execution_results(pred_exec_results, current_gold_results)
                elif current_gold_results is None and pred_exec_results is not None: # Gold failed, pred succeeded
                    execution_match = False
                    if not eval_error_message: eval_error_message = "Gold SQL failed to execute, but predicted SQL succeeded."
                # If both pred and gold results are None (e.g. DML statements), and both succeeded, consider it a match for now.
                elif pred_exec.success and (current_gold_results is None and pred_exec_results is None):
                    # This logic needs to be careful for DML. Comparing rows_affected might be better.
                    # For now, if both are non-SELECT and succeed, assume execution match conceptually.
                    if not predicted_sql.strip().upper().startswith("SELECT") and \
                       not gold_sql.strip().upper().startswith("SELECT"):
                       # Could compare rows_affected if available from SQLExecutionResult
                       execution_match = True # Placeholder for DML match
                    else: # Should have results if SELECT
                       execution_match = False


            else: # Predicted SQL execution failed
                syntax_valid = False # Assume syntax error if execution fails. Could be other runtime errors.
                eval_error_message = f"Predicted SQL execution failed: {pred_exec.error_message}"
                logger.warning(eval_error_message)
                execution_match = False
        else:
            logger.info("No SQL executor. Skipping execution-based evaluation and syntax check for QID {question_id}.")

        return EvaluationResult(
            question_id=question_id,
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            exact_match=exact_match,
            execution_match=execution_match,
            syntax_valid=syntax_valid,
            predicted_results=pred_exec_results,
            gold_results=gold_results_data, # Or the ones fetched if not provided
            error_message=eval_error_message
        )

    def run_evaluation(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs evaluation over a list of prediction/gold pairs.

        Args:
            evaluation_data: A list of dictionaries, each containing:
                             {'id': Any, 'predicted_sql': str, 'gold_sql': str,
                              'gold_results': Optional[List[Dict]]}

        Returns:
            A dictionary summarizing evaluation metrics.
        """
        all_results: List[EvaluationResult] = []
        num_exact_matches = 0
        num_execution_matches = 0
        num_valid_syntax = 0
        total_evaluated = 0

        for item in evaluation_data:
            qid = item.get("id")
            pred_sql = item.get("predicted_sql")
            gold_sql = item.get("gold_sql")
            gold_res = item.get("gold_results") # Optional pre-fetched gold results

            if qid is None or pred_sql is None or gold_sql is None:
                logger.warning(f"Skipping evaluation item due to missing data: {item}")
                continue

            total_evaluated += 1
            eval_res = self.evaluate_one(qid, pred_sql, gold_sql, gold_results_data=gold_res)
            all_results.append(eval_res)

            if eval_res.exact_match:
                num_exact_matches += 1
            if eval_res.execution_match: # True if execution matches
                num_execution_matches += 1
            if eval_res.syntax_valid: # True if syntax is valid
                num_valid_syntax += 1

        summary_metrics = {
            "total_queries": total_evaluated,
            "exact_match_accuracy": (num_exact_matches / total_evaluated) if total_evaluated else 0,
            "execution_accuracy": (num_execution_matches / total_evaluated) if total_evaluated and self.sql_executor else "N/A (No Executor)",
            "valid_syntax_rate": (num_valid_syntax / total_evaluated) if total_evaluated and self.sql_executor else "N/A (No Executor)",
            "num_exact_matches": num_exact_matches,
            "num_execution_matches": num_execution_matches if self.sql_executor else "N/A",
            "num_valid_syntax": num_valid_syntax if self.sql_executor else "N/A",
        }

        logger.info(f"Evaluation Summary: {summary_metrics}")
        # Could also return all_results for detailed inspection
        return {"summary": summary_metrics, "details": all_results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Conceptual: Mock SQLExecutor for testing evaluator logic
    class MockSQLExecutor:
        def execute_sql(self, sql_query: str, params=None, is_script=False):
            logger.info(f"[MockSQLExecutor] Executing: {sql_query[:50]}...")
            if "error" in sql_query.lower():
                return {"success": False, "error_message": "Simulated SQL error"}
            if "customers" in sql_query.lower() and "name" in sql_query.lower():
                return {"success": True, "results": [{"name": "Alice"}, {"name": "Bob"}]}
            if "products" in sql_query.lower() and "price" in sql_query.lower():
                 return {"success": True, "results": [{"product_name": "Laptop", "price": 1200.00}]}
            return {"success": True, "results": [], "rows_affected": 0}

    mock_executor = MockSQLExecutor()
    evaluator = BenchmarkEvaluator(sql_executor=mock_executor)

    sample_eval_data = [
        {
            "id": "q1",
            "predicted_sql": "SELECT name FROM customers;",
            "gold_sql": "SELECT name FROM customers", # Test normalization
            # "gold_results": [{"name": "Alice"}, {"name": "Bob"}] # Optionally provide gold results
        },
        {
            "id": "q2",
            "predicted_sql": "SELECT price, product_name FROM products;", # Different column order
            "gold_sql": "SELECT product_name, price FROM products;",
            "gold_results": [{"product_name": "Laptop", "price": 1200.00}]
        },
        {
            "id": "q3", # Test execution mismatch
            "predicted_sql": "SELECT name FROM customers WHERE id = 1;", # Returns Alice, Bob
            "gold_sql": "SELECT name FROM customers WHERE id = 99;", # Returns empty conceptually
            "gold_results": []
        },
        {
            "id": "q4", # Test predicted SQL error
            "predicted_sql": "SELECT name FROM error_table;",
            "gold_sql": "SELECT name FROM customers;"
        }
    ]

    logger.info("\n--- Running Benchmark Evaluation (Conceptual) ---")
    evaluation_summary = evaluator.run_evaluation(sample_eval_data)

    print("\nEvaluation Summary:")
    for k, v in evaluation_summary["summary"].items():
        print(f"  {k}: {v}")

    # print("\nDetailed Results:")
    # for res_detail in evaluation_summary["details"]:
    #     print(f"  {res_detail}")

    # Example direct comparison
    print("\nDirect SQL comparison:")
    print(f"  Match? ('SELECT a FROM t' vs 'select a from t;'): {evaluator.compare_sql_queries('SELECT a FROM t', 'select a from t;')}") # True
    print(f"  Match? ('SELECT a,b FROM t' vs 'SELECT b, a FROM t'): {evaluator.compare_sql_queries('SELECT a,b FROM t', 'SELECT b, a FROM t')}") # False

    # Example results comparison
    print("\nDirect results comparison:")
    res1 = [{"colA": 1, "colB": "x"}, {"colA": 2, "colB": "y"}]
    res2 = [{"colB": "y", "colA": 2}, {"colB": "x", "colA": 1}] # Same content, different order of rows and keys
    print(f"  Match? (res1 vs res2): {evaluator.compare_execution_results(res1, res2)}") # True

    res3 = [{"colA": 1, "colB": "x"}]
    print(f"  Match? (res1 vs res3): {evaluator.compare_execution_results(res1, res3)}") # False

    logger.info("Conceptual benchmark_evaluator.py example finished.")

# Ensure __init__.py exists in src/evaluation/
# try:
#     (Path(__file__).parent / "__init__.py").touch(exist_ok=True)
# except NameError:
#     pass
# print("src/evaluation/benchmark_evaluator.py created.")
