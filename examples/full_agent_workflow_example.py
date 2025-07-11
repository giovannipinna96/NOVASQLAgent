# examples/full_agent_workflow_example.py

import sys
import os
import json

# Adjust the path to include the src directory
# This is important for the example to find the (dummy) modules if they were separate files.
# However, for this self-contained example, we'll redefine simplified dummy classes.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Simplified Dummy Modules for Integrated Workflow ---
# These are redefined here for a self-contained example.
# In a real scenario, you would import these from their respective files in src/
# e.g., from src.model.prompt_template_manager import PromptTemplateManager

class DummyPromptTemplateManager:
    def __init__(self):
        self.templates = {
            "text_to_sql": "Schema: {{db_schema}}\nQuestion: {{nl_question}}\nSQL:",
            "evaluation_prompt": "NLQ: {{nl_question}}\nGenerated SQL: {{generated_sql}}\nIs it correct (Yes/No/Partial)? Why? {{error_if_any}}"
        }
        print("DummyPromptTemplateManager (Integrated) initialized.")

    def format_prompt(self, template_name, **kwargs):
        template = self.templates.get(template_name, "")
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        print(f"Formatted Prompt ('{template_name}'):\n{template}")
        return template

class DummyLLMmodel:
    def __init__(self, model_name="dummy_translator_model"):
        self.model_name = model_name
        print(f"DummyLLMmodel (Integrated - {model_name}) initialized.")

    def generate_text(self, prompt, **kwargs): # Generic text generation
        print(f"DummyLLMmodel ({self.model_name}) received prompt (first 50 chars): {prompt[:50]}...")
        if "Schema:" in prompt and "Question:" in prompt and "SQL:" in prompt: # Specific for Text-to-SQL
            if "list all users from California" in prompt.lower():
                response = "SELECT user_id, name, email FROM users WHERE state = 'California';"
            elif "total orders for product 'Gadget Alpha'" in prompt.lower():
                response = "SELECT SUM(oi.quantity * p.price) AS total_sales FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.name = 'Gadget Alpha';"
            else:
                response = "SELECT dummy_column FROM dummy_table WHERE condition;"
            print(f"Simulated SQL from LLM: {response}")
            return response
        elif "Is it correct" in prompt: # Specific for Judge
             if "SELECT user_id, name, email FROM users WHERE state = 'California';" in prompt and "error" not in prompt.lower():
                 return "Yes. The SQL query correctly selects users from California."
             elif "syntax error" in prompt.lower():
                 return "No. The SQL has a syntax error near 'FROMM'."
             return "Partial. Query is okay but could be optimized."
        return f"Generic LLM response based on model {self.model_name}."

class DummySQLSandbox:
    def __init__(self, db_config=None):
        self.db_config = db_config or {"type": "dummy_sqlite"}
        self._dummy_schema_str = "Table users: user_id (INT), name (TEXT), email (TEXT), state (TEXT)\nTable products: product_id (INT), name (TEXT), price (DECIMAL)\nTable order_items: order_id (INT), product_id (INT), quantity (INT)"
        print(f"DummySQLSandbox (Integrated) initialized with config: {self.db_config}")

    def get_schema_info_str(self): # Simplified to return string directly
        print(f"DummySQLSandbox: Providing schema string: {self._dummy_schema_str}")
        return self._dummy_schema_str

    def execute_sql(self, sql_query, read_only=True):
        print(f"DummySQLSandbox: Executing SQL (read_only={read_only}): {sql_query}")
        if "SELECT user_id, name, email FROM users WHERE state = 'California';" in sql_query:
            return {"status": "success", "columns": ["user_id", "name", "email"], "rows": [{"user_id": 1, "name": "Alice", "email": "alice@ca.com"}, {"user_id": 5, "name": "Bob", "email": "bob@ca.com"}], "row_count": 2}
        elif "SUM(oi.quantity * p.price)" in sql_query:
             return {"status": "success", "columns": ["total_sales"], "rows": [{"total_sales": 1250.75}], "row_count": 1}
        elif "FROMM" in sql_query: # Simulate syntax error
            return {"status": "error", "message": "Syntax error near 'FROMM'", "error_code": "SQL101"}
        elif "DROP TABLE" in sql_query.upper() and read_only:
            return {"status": "error", "message": "Attempt to DROP TABLE in read-only mode."}
        return {"status": "success", "columns": ["message"], "rows": [{"message": "Query executed (dummy data)."}], "row_count": 1}

class DummyLLMasJudge:
    def __init__(self, judge_llm_model): # Takes an LLM model instance
        self.judge_llm = judge_llm_model
        print("DummyLLMasJudge (Integrated) initialized.")

    def evaluate_sql_execution(self, nl_question, generated_sql, execution_result, prompt_template_manager):
        print(f"DummyLLMasJudge: Evaluating SQL execution for: {nl_question}")
        error_info = ""
        if execution_result.get("status") == "error":
            error_info = f"Execution Error: {execution_result.get('message')}"

        eval_prompt = prompt_template_manager.format_prompt(
            template_name="evaluation_prompt",
            nl_question=nl_question,
            generated_sql=generated_sql,
            error_if_any=error_info
        )
        feedback = self.judge_llm.generate_text(eval_prompt) # Use the LLM to generate feedback
        print(f"Judge Feedback: {feedback}")
        # Simplified score based on feedback
        score = 0.0
        if "Yes." in feedback: score = 1.0
        elif "Partial." in feedback: score = 0.5
        elif execution_result.get("status") == "success" and "No." not in feedback : score = 0.7 # Executed but LLM might have other concerns

        return {"feedback": feedback, "score": score, "execution_status": execution_result.get("status")}

# --- Main Workflow ---
def run_full_agent_workflow(natural_language_query):
    print(f"\n--- Starting Full Agent Workflow for Query: '{natural_language_query}' ---")

    # 1. Initialize components
    print("\n[Step 1: Initializing Components]")
    prompt_manager = DummyPromptTemplateManager()
    # One LLM for translation, another (or same) for judgment
    translator_llm = DummyLLMmodel(model_name="text-to-sql-llm")
    judge_llm = DummyLLMmodel(model_name="critique-llm") # Could be the same instance or a different one
    sql_sandbox = DummySQLSandbox()
    evaluator = DummyLLMasJudge(judge_llm_model=judge_llm)

    # 2. Get Database Schema (from SQL Sandbox)
    print("\n[Step 2: Retrieving Database Schema]")
    db_schema_str = sql_sandbox.get_schema_info_str()

    # 3. Generate SQL Query (using PromptManager and TranslatorLLM)
    print("\n[Step 3: Generating SQL Query]")
    sql_generation_prompt = prompt_manager.format_prompt(
        template_name="text_to_sql",
        db_schema=db_schema_str,
        nl_question=natural_language_query
    )
    generated_sql = translator_llm.generate_text(sql_generation_prompt)
    if not generated_sql or not generated_sql.strip().upper().startswith("SELECT"): # Basic check
        print("LLM failed to generate a valid SQL query. Exiting workflow.")
        return

    # 4. Execute SQL Query (using SQLSandbox)
    print("\n[Step 4: Executing Generated SQL Query]")
    # For this example, let's assume SELECT queries are intended to be read-only.
    # More complex logic would determine if read_only should be False.
    execution_result = sql_sandbox.execute_sql(generated_sql, read_only=True)
    print(f"Execution Result: {execution_result}")

    # 5. Evaluate SQL and Execution (using LLMasJudge)
    print("\n[Step 5: Evaluating SQL Query and Execution]")
    evaluation = evaluator.evaluate_sql_execution(
        nl_question=natural_language_query,
        generated_sql=generated_sql,
        execution_result=execution_result,
        prompt_template_manager=prompt_manager # Judge might use its own prompts
    )
    print(f"Final Evaluation: Score={evaluation['score']:.2f}, Feedback='{evaluation['feedback']}'")

    # 6. Present Results
    print("\n[Step 6: Presenting Results]")
    print(f"Natural Language Query: {natural_language_query}")
    print(f"Generated SQL: {generated_sql}")
    if execution_result['status'] == 'success':
        print(f"Execution Data: {execution_result.get('row_count')} rows. Columns: {execution_result.get('columns')}")
        # print("Sample Row:", execution_result.get('rows')[0] if execution_result.get('rows') else "N/A")
    else:
        print(f"Execution Error: {execution_result.get('message')}")
    print(f"Agent's Evaluation Score: {evaluation['score']:.2f}")
    print(f"Agent's Feedback: {evaluation['feedback']}")

    print("\n--- Full Agent Workflow Complete ---")
    return {"generated_sql": generated_sql, "execution": execution_result, "evaluation": evaluation}


if __name__ == "__main__":
    print("===== Full Agent Workflow Example =====")

    # Example Query 1: Simple valid query
    query1 = "List all users from California"
    run_full_agent_workflow(query1)

    print("\n" + "="*40 + "\n")

    # Example Query 2: A query that might involve aggregation
    query2 = "What is the total orders for product 'Gadget Alpha'?"
    run_full_agent_workflow(query2)

    print("\n" + "="*40 + "\n")

    # Example Query 3: A query that the dummy LLM might generate with a syntax error
    # To make this happen, we'd need the LLM to produce "SELECT * FROMM users"
    # Our current dummy LLM is a bit too rule-based for that, but let's imagine it could.
    # For now, we'll use a query that our dummy SQL sandbox can specifically make an error for.
    # Let's assume the LLM (incorrectly) generates SQL with "FROMM"
    # We can simulate this by manually setting a "bad" SQL and skipping LLM generation for this specific test.

    print("Simulating a workflow where LLM generates bad SQL for Query 3:")
    print("\n--- Starting Full Agent Workflow for Query: 'Show all customers (simulating bad SQL gen)' ---")
    bad_sql_query = "SELECT * FROMM customers;" # Manually set bad SQL

    # Re-run parts of the workflow with this bad_sql
    print("\n[Step 1 & 2: Components Initialized, Schema Retrieved (Assumed)]")
    prompt_manager_bad = DummyPromptTemplateManager()
    judge_llm_bad = DummyLLMmodel(model_name="critique-llm")
    sql_sandbox_bad = DummySQLSandbox()
    evaluator_bad = DummyLLMasJudge(judge_llm_model=judge_llm_bad)

    print(f"\n[Step 3: Generated SQL Query (Manually set for test)]\n{bad_sql_query}")

    print("\n[Step 4: Executing Generated SQL Query]")
    execution_result_bad = sql_sandbox_bad.execute_sql(bad_sql_query, read_only=True)
    print(f"Execution Result: {execution_result_bad}")

    print("\n[Step 5: Evaluating SQL Query and Execution]")
    evaluation_bad = evaluator_bad.evaluate_sql_execution(
        nl_question="Show all customers (simulating bad SQL gen)",
        generated_sql=bad_sql_query,
        execution_result=execution_result_bad,
        prompt_template_manager=prompt_manager_bad
    )
    print(f"Final Evaluation: Score={evaluation_bad['score']:.2f}, Feedback='{evaluation_bad['feedback']}'")
    print("--- Full Agent Workflow (Bad SQL Simulation) Complete ---")

    print("\n===== Full Agent Workflow Example Finished =====")
