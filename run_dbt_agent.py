"""
Main execution script for the NovaSQLAgent targeting the Spider2-DBT benchmark.

This script orchestrates the agent's components:
1. Loads configuration.
2. Initializes the planner, LLM interface, tools (including DB schema inspector, SQL executor).
3. Takes a natural language question as input (or processes a batch from a benchmark file).
4. Invokes the planner to generate a plan.
5. Executes the plan:
    - Grounding NL to schema elements.
    - SQL generation (potentially iterative with refinement).
    - SQL execution against the target database.
6. Evaluates the generated SQL and execution results against gold standards.
7. Outputs results and metrics.

This is a conceptual outline. Actual implementation will require detailed integration
of all agent components.
"""
import logging
import argparse
from pathlib import Path

# Import agent components (adjust paths based on final structure)
# Assuming components are in 'src' and this script is at root.
# Need to ensure PYTHONPATH includes 'src' or use relative imports if run as a module.
# For simplicity, let's assume src is on PYTHONPATH or this script handles path adjustments.
try:
    from src.config.agent_config import load_agent_settings, AgentSettings
    from src.agent.planner import Planner
    from src.grounding.grounder import Grounder
    from src.llm.llm_model import BaseLLM, HuggingFaceLLM, OpenAILLM # For type hinting and potential direct use
    from src.llm.langchain_llm import LangChainLLM # Example of another LLM interface
    from src.execution.sql_executor import SQLExecutor
    from src.evaluation.benchmark_evaluator import BenchmarkEvaluator
    # from src.tools.schema_inspector import SchemaInspectorTool # Example tool
    # from src.tools.sql_sandbox import SQLSandbox # Used by SQLExecutor
except ImportError as e:
    print(f"Error importing agent components: {e}. Ensure 'src' is in PYTHONPATH or script is run appropriately.")
    # Define placeholders if imports fail, to allow script structure to be parsed.
    class AgentSettings: pass # type: ignore
    class Planner: pass # type: ignore
    class Grounder: pass # type: ignore
    class BaseLLM: pass # type: ignore
    class HuggingFaceLLM: pass # type: ignore
    class OpenAILLM: pass # type: ignore
    class LangChainLLM: pass # type: ignore
    class SQLExecutor: pass # type: ignore
    class BenchmarkEvaluator: pass # type: ignore
    def load_agent_settings(config_path=None): return AgentSettings() #type: ignore


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DBTAgentRunner")

def get_llm_instance(llm_config_name: str, settings: AgentSettings) -> Optional[BaseLLM]:
    """Helper function to initialize an LLM instance based on config."""
    config = settings.get_llm_config(llm_config_name)
    if not config:
        logger.error(f"LLM configuration '{llm_config_name}' not found in settings.")
        return None

    logger.info(f"Initializing LLM '{config.model_name}' from provider '{config.provider}'.")

    # Common parameters for LLM model initialization
    common_params = {
        "model_name": config.model_name,
        "task": "sql_generation", # Example task
        # prompt_template_path can be part of LLM model's own config or passed if needed
        # use_memory and agent_id also part of LLM model config conceptually
    }

    if config.provider == "huggingface":
        # Ensure HuggingFaceLLM and its dependencies are available
        if not globals().get("HuggingFaceLLM") or HuggingFaceLLM is BaseLLM : # Check if it's the placeholder
            logger.error("HuggingFaceLLM class not available/imported correctly.")
            return None
        try:
            return HuggingFaceLLM(
                **common_params,
                use_4bit=config.use_4bit,
                use_8bit=config.use_8bit,
                device=config.hf_device
                # model_kwargs, tokenizer_kwargs can be added from config if defined
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceLLM for {config.model_name}: {e}")
            return None
    elif config.provider == "openai":
        if not globals().get("OpenAILLM") or OpenAILLM is BaseLLM:
            logger.error("OpenAILLM class not available/imported correctly.")
            return None
        try:
            return OpenAILLM(
                **common_params,
                api_key=config.api_key
                # api_kwargs can be added
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAILLM for {config.model_name}: {e}")
            return None
    # Add other providers like Anthropic, LangChain wrappers, etc.
    # Example for LangChainLLM (if it were a primary choice)
    # elif config.provider == "langchain_hf_custom":
    #     if not globals().get("LangChainLLM") or LangChainLLM is BaseLLM: # type: ignore
    #          logger.error("LangChainLLM class not available/imported correctly.")
    #          return None
    #     try:
    #         return LangChainLLM( # type: ignore
    #             model_name_or_path=config.model_name,
    #             device=config.hf_device,
    #             use_peft=False, # Example, could be in config
    #             max_new_tokens=config.max_new_tokens,
    #             temperature=config.temperature
    #         )
    #     except Exception as e:
    #          logger.error(f"Failed to initialize LangChainLLM for {config.model_name}: {e}")

    else:
        logger.error(f"Unsupported LLM provider: {config.provider}")
        return None


def main_agent_run(question: str, db_id: str, agent_settings: AgentSettings) -> Optional[Dict[str, Any]]:
    """
    Main logic for running the agent for a single question and database.
    This is a high-level conceptual flow.
    """
    logger.info(f"Processing question for database '{db_id}': '{question}'")

    # 1. Initialize components based on settings
    # LLM
    llm = get_llm_instance(agent_settings.default_llm, agent_settings)
    if not llm:
        logger.error("Failed to initialize LLM. Aborting run for this question.")
        return {"error": "LLM initialization failed", "predicted_sql": None}

    logger.info(f"Using LLM: {llm.model_name} ({llm.__class__.__name__})")

    # Database Executor
    db_config = agent_settings.get_db_config(db_id) # db_id might be a specific name or type
    if not db_config:
        # Fallback: try to find a generic SQLite config if db_id is not in target_databases
        db_config = agent_settings.get_db_config("sqlite")
        if db_config:
            logger.warning(f"DB config for '{db_id}' not found, using generic SQLite config with path '{db_config.db_path}'. This might be incorrect for the target DB.")
            # If db_id is a path to a specific sqlite db for the benchmark, update db_path
            if Path(db_id).suffix == ".sqlite" and db_config.db_type == "sqlite":
                db_config.db_path = db_id
        else:
            logger.error(f"Database configuration for '{db_id}' not found. Aborting.")
            return {"error": f"DB config for '{db_id}' not found", "predicted_sql": None}

    if not globals().get("SQLExecutor"):
        logger.error("SQLExecutor class not available.")
        return {"error": "SQLExecutor not available", "predicted_sql": None}

    sql_executor = SQLExecutor(db_type=db_config.db_type, db_config=db_config.__dict__ if hasattr(db_config, '__dict__') else {}) # Pass config as dict

    # Planner, Grounder, Tools
    # These would be initialized here, potentially using the LLM or DB info.
    # planner = Planner(llm_interface=llm)
    # grounder = Grounder(llm_interface=llm, db_schema_info=...) # Schema needs to be loaded
    # schema_inspector = SchemaInspectorTool(sql_executor=sql_executor) # Example tool

    # Conceptual: Get DB Schema for the current db_id
    # This would involve using a tool or specific logic for Spider2-DBT benchmark data.
    # For now, assume schema is available as a dictionary.
    # db_schema_dict = schema_inspector.get_schema_for_db(db_id) # Conceptual
    db_schema_dict = {"tables": {"example_table": {"columns": ["id", "name"]}}, "db_id": db_id} # Placeholder
    logger.info(f"Conceptual schema for {db_id}: {str(db_schema_dict)[:200]}...")


    # 2. Agent Workflow (Conceptual)
    #    a. Plan generation (if using a planner)
    #    plan = planner.generate_plan(question, db_schema_dict)
    #    logger.info(f"Generated plan: {plan}")

    #    b. Grounding (if part of the plan or a default step)
    #    groundings = grounder.ground_question(question, db_schema_dict)
    #    logger.info(f"Groundings: {groundings}")

    #    c. SQL Generation (Iterative or single-shot using LLM)
    #       This is the core part. The prompt to the LLM would include the question,
    #       schema, groundings (if any), and instructions to generate SQL.
    prompt_for_sql = (
        f"Database Schema for '{db_id}':\n"
        f"{json.dumps(db_schema_dict, indent=2)}\n\n"
        f"User Question: {question}\n\n"
        f"Generate the SQL query to answer the question based on the schema.\n"
        f"SQL Query:"
    )

    logger.debug(f"Prompt for SQL generation:\n{prompt_for_sql}")

    try:
        # Using the LLM's run method directly for this conceptual step
        predicted_sql = llm.run(prompt_for_sql, generation_kwargs={"max_new_tokens": 256}) # Adjust kwargs as needed
        logger.info(f"LLM generated SQL: {predicted_sql}")
    except Exception as e:
        logger.error(f"Error during SQL generation by LLM: {e}", exc_info=True)
        return {"error": f"LLM SQL generation failed: {e}", "predicted_sql": None}

    #    d. (Optional) SQL Refinement / Validation (e.g., syntax check, self-correction loop)
    #       This could involve another LLM call or rule-based checks.

    #    e. SQL Execution (using SQLExecutor)
    #       Only execute if evaluation requires it or if agent needs results for multi-turn.
    #       For benchmark, execution is often part of the evaluation phase.
    #       Let's assume for now we just return the SQL.
    #       Execution result can be obtained via:
    #       exec_result = sql_executor.execute_sql(predicted_sql)
    #       logger.info(f"Execution result of predicted SQL: {exec_result}")

    return {"predicted_sql": predicted_sql, "db_id": db_id, "question": question}


def run_benchmark(settings_file: str, benchmark_file: Optional[str] = None, single_question: Optional[str] = None, target_db_id: Optional[str] = None):
    """
    Runs the agent over a benchmark file or a single question.
    """
    agent_settings = load_agent_settings(settings_file)
    if not agent_settings:
        logger.error("Could not load agent settings. Exiting.")
        return

    logger.setLevel(agent_settings.log_level.upper()) # Set log level from config

    if single_question and target_db_id:
        logger.info(f"Running for single question on DB '{target_db_id}'.")
        result = main_agent_run(single_question, target_db_id, agent_settings)
        if result:
            print("\n--- Single Question Result ---")
            print(f"  Question: {result.get('question')}")
            print(f"  DB ID: {result.get('db_id')}")
            print(f"  Predicted SQL: {result.get('predicted_sql')}")
            if result.get('error'):
                print(f"  Error: {result.get('error')}")
        else:
            print("Agent did not produce a result for the single question.")

    elif benchmark_file:
        logger.info(f"Running benchmark from file: {benchmark_file}")
        # Conceptual: Load benchmark data (list of questions, db_ids, gold_sqls)
        # Example structure of benchmark_data_item:
        # {"id": "q1", "question": "...", "db_id": "...", "gold_sql": "..."}

        # This is a placeholder for actual benchmark file loading
        try:
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f) # Assuming JSON list of dicts
            if not isinstance(benchmark_data, list):
                logger.error(f"Benchmark file {benchmark_file} should contain a JSON list.")
                return
        except Exception as e:
            logger.error(f"Failed to load or parse benchmark file {benchmark_file}: {e}")
            return

        all_predictions = []
        for item in benchmark_data:
            q = item.get("question")
            db = item.get("db_id")
            if not q or not db:
                logger.warning(f"Skipping benchmark item due to missing question or db_id: {item}")
                continue

            run_output = main_agent_run(q, db, agent_settings)
            prediction_record = {
                "id": item.get("id", db + "_" + q[:20]), # Create an ID if not present
                "db_id": db,
                "question": q,
                "predicted_sql": run_output.get("predicted_sql") if run_output else None,
                "gold_sql": item.get("gold_sql"), # For later evaluation
                "error": run_output.get("error") if run_output else None
            }
            all_predictions.append(prediction_record)
            logger.info(f"Completed processing for item ID {prediction_record['id']}.")
            # Small delay if hitting APIs rapidly
            # import time; time.sleep(0.5)


        # Save predictions to a file
        predictions_file = Path(agent_settings.evaluation_output_dir) / "predictions.json"
        agent_settings.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
        with open(predictions_file, "w") as f:
            json.dump(all_predictions, f, indent=2)
        logger.info(f"All predictions saved to {predictions_file}")

        # Conceptual: Run evaluation using BenchmarkEvaluator
        # evaluator = BenchmarkEvaluator(sql_executor=...) # Needs an executor for exec accuracy
        # evaluation_summary = evaluator.run_evaluation(all_predictions)
        # logger.info(f"Benchmark Evaluation Summary: {evaluation_summary['summary']}")
        # eval_details_file = Path(agent_settings.evaluation_output_dir) / "evaluation_details.json"
        # with open(eval_details_file, "w") as f:
        #    json.dump(evaluation_summary, f, indent=2) # Save full summary and details
        # logger.info(f"Evaluation details saved to {eval_details_file}")

    else:
        logger.warning("No benchmark file or single question provided. Exiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NovaSQLAgent for Spider2-DBT.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/agent_default_config.json", # Default path to agent settings
        help="Path to the agent configuration JSON file."
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        help="Path to the JSON file containing benchmark questions and database IDs."
    )
    parser.add_argument(
        "--question",
        type=str,
        help="A single natural language question to process."
    )
    parser.add_argument(
        "--db_id",
        type=str,
        help="The database ID to target for the single question (e.g., 'customers_campaigns', 'concert_singer'). For SQLite, can be path to .sqlite file."
    )

    args = parser.parse_args()

    if args.question and not args.db_id:
        parser.error("--db_id is required if --question is provided.")
    if not args.question and args.db_id:
        parser.error("--question is required if --db_id is provided.")

    if not args.benchmark_file and not (args.question and args.db_id):
        logger.info("No benchmark file or single question specified. Running conceptual self-test with default config.")
        # Conceptual self-test (does not run full agent logic, just tests config loading)
        # This part is illustrative. A real self-test would involve a dummy question.
        default_settings = load_agent_settings(args.config)
        if default_settings:
            logger.info(f"Successfully loaded default settings for agent '{default_settings.agent_name}'.")
            llm_conf = default_settings.get_llm_config()
            if llm_conf:
                 logger.info(f"Default LLM configured: {llm_conf.model_name} by {llm_conf.provider}")
            else:
                 logger.error("Default LLM configuration could not be loaded.")
        else:
            logger.error("Failed to load default agent settings.")
        logger.info("To run on data, provide --benchmark_file or --question and --db_id.")
    else:
        run_benchmark(
            settings_file=args.config,
            benchmark_file=args.benchmark_file,
            single_question=args.question,
            target_db_id=args.db_id
        )

    logger.info("run_dbt_agent.py finished.")

# Placeholder for imports if they fail initially due to path issues
import json # Already imported, but good for structure
# print("run_dbt_agent.py created at project root.")
