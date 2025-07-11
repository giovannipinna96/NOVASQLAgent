# examples/model_LLMasJudge_example.py

import sys
import os

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.model.LLMasJudge import LLMasJudge
    # from src.model.LLMasJudge import another_utility_if_any
except ImportError:
    print("Warning: Could not import LLMasJudge from src.model.LLMasJudge.")
    print("Using a dummy LLMasJudge class for demonstration purposes.")

    class LLMasJudge:
        """
        Dummy LLMasJudge class for demonstration if the actual module/class
        cannot be imported or its structure is unknown.
        """
        def __init__(self, judge_model_name="gpt-4-judge", config=None):
            self.judge_model_name = judge_model_name
            self.config = config if config else {}
            print(f"Dummy LLMasJudge initialized with model: {self.judge_model_name}, Config: {self.config}")

        def evaluate_sql_query(self, nl_question, generated_sql, reference_sql=None, db_schema=None):
            """
            Dummy method to simulate evaluating a generated SQL query.
            """
            print(f"\nEvaluating SQL Query for: '{nl_question}'")
            print(f"Generated SQL: {generated_sql}")
            if reference_sql:
                print(f"Reference SQL: {reference_sql}")
            if db_schema:
                print(f"DB Schema context: {db_schema}")

            # Simulate evaluation logic
            score = 0.85  # Dummy score
            feedback = "The query seems plausible. Consider adding table aliases for very complex joins."
            if reference_sql and generated_sql.strip().lower() == reference_sql.strip().lower():
                score = 1.0
                feedback = "Perfect match with reference SQL."
            elif "DROP TABLE" in generated_sql.upper():
                score = 0.1
                feedback = "Potential destructive command detected."

            evaluation_result = {"score": score, "feedback": feedback, "is_correct_syntax": True}
            print(f"Evaluation Result (dummy): {evaluation_result}")
            return evaluation_result

        def score_text_response(self, question, response, criteria, reference_answer=None):
            """
            Dummy method to simulate scoring a textual response based on criteria.
            """
            print(f"\nScoring Text Response for: '{question}'")
            print(f"Response: '{response}'")
            print(f"Criteria: {criteria}")
            if reference_answer:
                print(f"Reference Answer: '{reference_answer}'")

            # Simulate scoring
            score = 0.7
            remarks = "Response is relevant but lacks depth in addressing criterion 'detail'."
            if "excellent" in response.lower() and "detail" in criteria:
                 score = 0.9
                 remarks = "Response is excellent and detailed."


            scoring_result = {"overall_score": score, "remarks_per_criterion": {"relevance": 0.8, "clarity": 0.7, "detail": (0.6 if "detail" in criteria else "N/A")}, "detailed_feedback": remarks}
            print(f"Scoring Result (dummy): {scoring_result}")
            return scoring_result

        def compare_responses(self, question, response_A, response_B, criteria):
            """
            Dummy method to compare two responses and pick the better one.
            """
            print(f"\nComparing two responses for question: '{question}'")
            print(f"Response A: '{response_A}'")
            print(f"Response B: '{response_B}'")
            print(f"Comparison Criteria: {criteria}")

            # Simulate comparison
            preference = "Response_A"
            justification = "Response A is more concise and directly answers the question based on the given criteria."
            if len(response_B) > len(response_A) and "conciseness" not in criteria: # Arbitrary logic
                preference = "Response_B"
                justification = "Response B provides more information."

            comparison_outcome = {"preferred_response": preference, "justification": justification}
            print(f"Comparison Outcome (dummy): {comparison_outcome}")
            return comparison_outcome

def main():
    print("--- LLMasJudge Module Example ---")

    # Instantiate LLMasJudge, possibly with a specific judge model or configuration
    try:
        judge_config = {"api_key": "your_judge_llm_api_key", "timeout": 60}
        llm_judge = LLMasJudge(judge_model_name="claude-judge-v1", config=judge_config)
    except NameError: # Fallback for dummy
        judge_config = {"api_key": "DUMMY_JUDGE_KEY", "timeout": 60}
        llm_judge = LLMasJudge(judge_model_name="dummy-judge-v1", config=judge_config)


    # Example 1: Evaluate a generated SQL query
    print("\n[Example 1: Evaluate SQL Query]")
    nl_q = "Find all customers from California"
    gen_sql = "SELECT name, email FROM customers WHERE state = 'CA';"
    ref_sql = "SELECT name, email_address FROM customers WHERE state_province = 'CA';" # Slightly different
    db_schema_info = {"tables": ["customers (customer_id, name, email_address, state_province)"]}

    evaluation = llm_judge.evaluate_sql_query(
        nl_question=nl_q,
        generated_sql=gen_sql,
        reference_sql=ref_sql,
        db_schema=db_schema_info
    )
    print(f"Evaluation for '{gen_sql}': Score={evaluation['score']}, Feedback='{evaluation['feedback']}'")

    # Example 2: Evaluate another SQL query (potentially problematic)
    print("\n[Example 2: Evaluate Problematic SQL Query]")
    gen_sql_problematic = "SELECT * FROM users; DROP TABLE orders; --"
    evaluation_problematic = llm_judge.evaluate_sql_query(
        nl_question="Show all users and then remove orders",
        generated_sql=gen_sql_problematic,
        db_schema=db_schema_info  # schema might be used to detect issues
    )
    print(f"Evaluation for '{gen_sql_problematic}': Score={evaluation_problematic['score']}, Feedback='{evaluation_problematic['feedback']}'")


    # Example 3: Score a textual response
    if hasattr(llm_judge, "score_text_response"):
        print("\n[Example 3: Score Text Response]")
        question_text = "Explain the concept of photosynthesis."
        response_text = "Photosynthesis is a process used by plants to convert light energy into chemical energy."
        scoring_criteria = ["accuracy", "completeness", "clarity"]
        reference_text = "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose (a sugar). This process uses carbon dioxide and water, and releases oxygen as a byproduct."

        score_details = llm_judge.score_text_response(
            question=question_text,
            response=response_text,
            criteria=scoring_criteria,
            reference_answer=reference_text
        )
        print(f"Text Response Score: {score_details['overall_score']}, Feedback: {score_details['detailed_feedback']}")

    # Example 4: Compare two responses
    if hasattr(llm_judge, "compare_responses"):
        print("\n[Example 4: Compare Two Text Responses]")
        comp_question = "What is the capital of France?"
        response1 = "The capital of France is Paris."
        response2 = "Paris, located on the river Seine, is the renowned capital of France, known for its art, fashion, and culture."
        comp_criteria = ["correctness", "conciseness"]

        comparison_result = llm_judge.compare_responses(
            question=comp_question,
            response_A=response1,
            response_B=response2,
            criteria=comp_criteria
        )
        print(f"Preferred Response: {comparison_result['preferred_response']}, Justification: {comparison_result['justification']}")

    print("\n--- LLMasJudge Module Example Complete ---")

if __name__ == "__main__":
    main()
