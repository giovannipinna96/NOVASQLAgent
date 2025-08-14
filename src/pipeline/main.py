"""Main pipeline orchestrator for the SQL generation pipeline.

This module contains the main SQLGenerationPipeline class that coordinates
all pipeline steps and agents to execute the complete workflow.
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .models.pipeline_config import PipelineConfig
from .models.pipeline_state import PipelineState, PipelineStatus
from .models.result_models import PipelineResult

from .agents.configuration_agent import ConfigurationAgent
from .agents.prompt_variant_agent import PromptVariantAgent
from .agents.relevance_evaluation_agent import RelevanceEvaluationAgent
from .agents.sql_generation_agent import SQLGenerationAgent
from .agents.query_merge_agent import QueryMergeAgent
from .agents.translation_agent import TranslationAgent
from .agents.output_agent import OutputAgent


class SQLGenerationPipeline:
    """
    Main orchestrator for the SQL generation pipeline.
    
    This class coordinates all pipeline steps using specialized agents
    to transform external prompts into final SQL queries.
    
    Pipeline workflow:
    1. Configuration management
    2. Prompt variant generation
    3. Relevance evaluation
    4. SQL generation
    5. Query merging (if needed)
    6. SQL translation (if specified)
    7. Output generation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, 
                 model_id: str = "microsoft/phi-4-mini-instruct"):
        """
        Initialize the SQL generation pipeline.
        
        Args:
            config: Pipeline configuration (defaults to PipelineConfig())
            model_id: HuggingFace model ID for agents
        """
        self.config = config or PipelineConfig()
        self.model_id = model_id
        self.state = None
        
        # Initialize agents
        self.agents = {
            "configuration": ConfigurationAgent(model_id),
            "prompt_variant": PromptVariantAgent(model_id),
            "relevance_evaluation": RelevanceEvaluationAgent(model_id),
            "sql_generation": SQLGenerationAgent(model_id),
            "query_merge": QueryMergeAgent(model_id),
            "translation": TranslationAgent(model_id),
            "output": OutputAgent(model_id)
        }
        
        self.execution_log = []
    
    def run(self, external_prompt: str, external_descriptions: List[str], 
            output_file: Optional[str] = None, config_overrides: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the complete SQL generation pipeline.
        
        Args:
            external_prompt: The original prompt for SQL generation
            external_descriptions: List of external descriptions to evaluate against
            output_file: Optional output file path for the final SQL
            config_overrides: Optional configuration overrides
            
        Returns:
            PipelineResult containing execution results and metadata
        """
        try:
            # Initialize pipeline state
            self.state = PipelineState(
                external_prompt=external_prompt,
                external_descriptions=external_descriptions,
                pipeline_name=self.config.pipeline_name
            )
            self.state.status = PipelineStatus.RUNNING
            
            # Apply configuration overrides
            if config_overrides:
                self._apply_config_overrides(config_overrides)
            
            # Override output file if specified
            if output_file:
                self.config.output.output_file = output_file
            
            self._log_info("Starting SQL generation pipeline")
            
            # Execute pipeline steps
            steps_config = self._prepare_steps_config()
            
            # Step 1: Configuration management
            if not self._execute_configuration_step(steps_config["configuration"]):
                return self._create_failed_result("Configuration step failed")
            
            # Step 2: Prompt variant generation
            if not self._execute_prompt_variant_step(steps_config["prompt_variant"]):
                return self._create_failed_result("Prompt variant generation failed")
            
            # Step 3: Relevance evaluation
            if not self._execute_relevance_evaluation_step(steps_config["relevance_evaluation"]):
                return self._create_failed_result("Relevance evaluation failed")
            
            # Step 4: SQL generation
            if not self._execute_sql_generation_step(steps_config["sql_generation"]):
                return self._create_failed_result("SQL generation failed")
            
            # Step 5: Query merging (if needed)
            if not self._execute_query_merge_step(steps_config["query_merge"]):
                return self._create_failed_result("Query merging failed")
            
            # Step 6: SQL translation (if specified)
            if not self._execute_translation_step(steps_config["translation"]):
                return self._create_failed_result("SQL translation failed")
            
            # Step 7: Output generation
            if not self._execute_output_step(steps_config["output"]):
                return self._create_failed_result("Output generation failed")
            
            # Complete pipeline
            self.state.complete_pipeline(PipelineStatus.COMPLETED)
            
            self._log_info(f"Pipeline completed successfully in {self.state.total_duration_seconds:.2f}s")
            
            # Create and return result
            return PipelineResult.from_pipeline_state(self.state)
            
        except Exception as e:
            # Handle unexpected errors
            self._log_error(f"Pipeline execution failed: {e}")
            
            if self.state:
                self.state.add_error("pipeline", str(e))
                self.state.complete_pipeline(PipelineStatus.FAILED)
                return PipelineResult.from_pipeline_state(self.state)
            else:
                # Create minimal error result
                return PipelineResult(
                    pipeline_id="error",
                    pipeline_name=self.config.pipeline_name,
                    execution_time=datetime.now(),
                    duration_seconds=0,
                    success=False,
                    errors=[{"source": "pipeline", "message": str(e)}]
                )
    
    def _prepare_steps_config(self) -> Dict[str, Dict[str, Any]]:
        """Prepare configuration for each pipeline step."""
        return {
            "configuration": self.config.configuration_step.__dict__,
            "prompt_variant": self.config.prompt_variant.__dict__,
            "relevance_evaluation": self.config.relevance_evaluation.__dict__,
            "sql_generation": self.config.sql_generation.__dict__,
            "query_merge": self.config.query_merge.__dict__,
            "translation": self.config.translation.__dict__,
            "output": self.config.output.__dict__
        }
    
    def _execute_configuration_step(self, config: Dict[str, Any]) -> bool:
        """Execute configuration management step."""
        try:
            step_result = self.state.start_step("configuration")
            
            input_data = {
                "config_file_path": config.get("config_file_path", "./pipeline_config.json"),
                "operation": "read"  # Try to read, create if not exists
            }
            
            result = self.agents["configuration"].run(input_data, config)
            
            if result.get("success", False):
                # Store configuration data in state
                output_data = result.get("output_data", {})
                self.state.configuration.config_data = output_data.get("config_data", {})
                self.state.configuration.config_file_path = output_data.get("config_file", "")
                self.state.configuration.config_file_read = True
                
                self.state.complete_step("configuration", output_data)
                return True
            else:
                error_msg = result.get("error_message", "Configuration step failed")
                self.state.fail_step("configuration", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("configuration", f"Configuration step error: {e}")
            return False
    
    def _execute_prompt_variant_step(self, config: Dict[str, Any]) -> bool:
        """Execute prompt variant generation step."""
        try:
            step_result = self.state.start_step("prompt_variant")
            
            input_data = {
                "original_prompt": self.state.external_prompt,
                "external_descriptions": self.state.external_descriptions
            }
            
            result = self.agents["prompt_variant"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store variants in state
                variants = output_data.get("variants", [])
                self.state.prompt_variants.original_prompt = self.state.external_prompt
                self.state.prompt_variants.variants = [v.get("text", "") for v in variants]
                self.state.prompt_variants.generation_metadata = output_data.get("generation_stats", {})
                
                self.state.complete_step("prompt_variant", output_data)
                return True
            else:
                error_msg = result.get("error_message", "Prompt variant generation failed")
                self.state.fail_step("prompt_variant", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("prompt_variant", f"Prompt variant step error: {e}")
            return False
    
    def _execute_relevance_evaluation_step(self, config: Dict[str, Any]) -> bool:
        """Execute relevance evaluation step."""
        try:
            step_result = self.state.start_step("relevance_evaluation")
            
            # Prepare variants data
            variant_objects = []
            for i, variant_text in enumerate(self.state.prompt_variants.variants):
                variant_objects.append({
                    "variant_id": i + 1,
                    "text": variant_text
                })
            
            input_data = {
                "prompt_variants": variant_objects,
                "external_descriptions": self.state.external_descriptions,
                "original_prompt": self.state.external_prompt
            }
            
            result = self.agents["relevance_evaluation"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store evaluation results in state
                self.state.relevance_evaluation.external_descriptions = self.state.external_descriptions
                self.state.relevance_evaluation.relevant_descriptions = output_data.get("relevant_descriptions", [])
                self.state.relevance_evaluation.evaluation_details = output_data.get("evaluation_details", {})
                
                self.state.complete_step("relevance_evaluation", output_data)
                return True
            else:
                error_msg = result.get("error_message", "Relevance evaluation failed")
                self.state.fail_step("relevance_evaluation", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("relevance_evaluation", f"Relevance evaluation step error: {e}")
            return False
    
    def _execute_sql_generation_step(self, config: Dict[str, Any]) -> bool:
        """Execute SQL generation step."""
        try:
            step_result = self.state.start_step("sql_generation")
            
            input_data = {
                "original_prompt": self.state.external_prompt,
                "relevant_descriptions": self.state.relevance_evaluation.relevant_descriptions,
                "prompt_variants": [{"text": v} for v in self.state.prompt_variants.variants]
            }
            
            # Add target dialect if specified
            if hasattr(self.config.sql_generation.model_config, 'target_dialect'):
                config['target_dialect'] = self.config.sql_generation.model_config.target_dialect
            
            result = self.agents["sql_generation"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store SQL generation results in state
                self.state.sql_generation.input_prompt = self.state.external_prompt
                self.state.sql_generation.relevant_context = self.state.relevance_evaluation.relevant_descriptions
                self.state.sql_generation.primary_query = output_data.get("primary_query", "")
                
                # Store alternative queries
                alternative_queries = output_data.get("alternative_queries", [])
                self.state.sql_generation.generated_queries = [
                    self.state.sql_generation.primary_query
                ] + [alt.get("sql_query", "") for alt in alternative_queries if alt.get("sql_query")]
                
                self.state.sql_generation.query_metadata = output_data.get("generation_metadata", {})
                
                # Store validation result
                validation_result = output_data.get("validation_result", {})
                self.state.sql_generation.syntax_validation = validation_result.get("valid", None)
                
                self.state.complete_step("sql_generation", output_data)
                return True
            else:
                error_msg = result.get("error_message", "SQL generation failed")
                self.state.fail_step("sql_generation", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("sql_generation", f"SQL generation step error: {e}")
            return False
    
    def _execute_query_merge_step(self, config: Dict[str, Any]) -> bool:
        """Execute query merging step."""
        try:
            step_result = self.state.start_step("query_merge")
            
            input_data = {
                "generated_queries": self.state.sql_generation.generated_queries,
                "primary_query": self.state.sql_generation.primary_query,
                "alternative_queries": []  # Will be populated from generated_queries
            }
            
            result = self.agents["query_merge"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store merge results in state
                self.state.query_merge.input_queries = self.state.sql_generation.generated_queries
                self.state.query_merge.merged_query = output_data.get("merged_query", "")
                self.state.query_merge.merge_strategy_used = output_data.get("merge_strategy", "")
                self.state.query_merge.merge_successful = output_data.get("merge_performed", False)
                self.state.query_merge.merge_metadata = output_data.get("merge_metadata", {})
                
                self.state.complete_step("query_merge", output_data)
                return True
            else:
                error_msg = result.get("error_message", "Query merging failed")
                self.state.fail_step("query_merge", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("query_merge", f"Query merge step error: {e}")
            return False
    
    def _execute_translation_step(self, config: Dict[str, Any]) -> bool:
        """Execute SQL translation step."""
        try:
            step_result = self.state.start_step("translation")
            
            # Use merged query if available, otherwise use primary query
            query_to_translate = (self.state.query_merge.merged_query if self.state.query_merge.merged_query 
                                else self.state.sql_generation.primary_query)
            
            input_data = {
                "sql_query": query_to_translate,
                "merged_query": self.state.query_merge.merged_query,
                "source_dialect": "generic"
            }
            
            result = self.agents["translation"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store translation results in state
                self.state.translation.input_query = query_to_translate
                self.state.translation.target_dialect = output_data.get("target_dialect", "")
                self.state.translation.source_dialect = output_data.get("source_dialect", "generic")
                self.state.translation.translated_query = output_data.get("translated_query", "")
                self.state.translation.translation_successful = output_data.get("translation_performed", False)
                self.state.translation.translation_metadata = output_data.get("translation_metadata", {})
                
                self.state.complete_step("translation", output_data)
                return True
            else:
                error_msg = result.get("error_message", "SQL translation failed")
                self.state.fail_step("translation", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("translation", f"Translation step error: {e}")
            return False
    
    def _execute_output_step(self, config: Dict[str, Any]) -> bool:
        """Execute output generation step."""
        try:
            step_result = self.state.start_step("output")
            
            # Determine final query (translated > merged > primary)
            final_query = (self.state.translation.translated_query if self.state.translation.translated_query
                          else self.state.query_merge.merged_query if self.state.query_merge.merged_query
                          else self.state.sql_generation.primary_query)
            
            input_data = {
                "final_query": final_query,
                "translated_query": self.state.translation.translated_query,
                "pipeline_metadata": self._create_pipeline_metadata(),
                "original_prompt": self.state.external_prompt
            }
            
            result = self.agents["output"].run(input_data, config)
            
            if result.get("success", False):
                output_data = result.get("output_data", {})
                
                # Store output results in state
                self.state.output.final_query = final_query
                self.state.output.output_file_path = output_data.get("output_file_path", "")
                self.state.output.backup_created = output_data.get("backup_created", False)
                self.state.output.metadata_written = output_data.get("metadata_written", False)
                self.state.output.output_successful = True
                
                self.state.complete_step("output", output_data)
                return True
            else:
                error_msg = result.get("error_message", "Output generation failed")
                self.state.fail_step("output", error_msg)
                return False
                
        except Exception as e:
            self.state.fail_step("output", f"Output step error: {e}")
            return False
    
    def _create_pipeline_metadata(self) -> Dict[str, Any]:
        """Create metadata about the pipeline execution."""
        return {
            "pipeline_info": {
                "pipeline_id": self.state.pipeline_id,
                "pipeline_name": self.state.pipeline_name,
                "start_time": self.state.start_time.isoformat(),
                "status": self.state.status.value,
                "steps_completed": len([r for r in self.state.step_results.values() if r.status.value == "completed"]),
                "total_duration": self.state.total_duration_seconds
            },
            "input_info": {
                "original_prompt": self.state.external_prompt,
                "external_descriptions_count": len(self.state.external_descriptions),
                "prompt_variants_count": len(self.state.prompt_variants.variants),
                "relevant_descriptions_count": len(self.state.relevance_evaluation.relevant_descriptions)
            },
            "processing_info": {
                "queries_generated": len(self.state.sql_generation.generated_queries),
                "merge_performed": self.state.query_merge.merge_successful,
                "translation_performed": self.state.translation.translation_successful,
                "syntax_validated": self.state.sql_generation.syntax_validation
            }
        }
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides."""
        try:
            for key, value in overrides.items():
                if hasattr(self.config, key):
                    if isinstance(getattr(self.config, key), dict):
                        getattr(self.config, key).update(value)
                    else:
                        setattr(self.config, key, value)
        except Exception as e:
            self._log_warning(f"Failed to apply config override for {key}: {e}")
    
    def _create_failed_result(self, error_message: str) -> PipelineResult:
        """Create a failed pipeline result."""
        if self.state:
            self.state.add_error("pipeline", error_message)
            self.state.complete_pipeline(PipelineStatus.FAILED)
            return PipelineResult.from_pipeline_state(self.state)
        else:
            return PipelineResult(
                pipeline_id="failed",
                pipeline_name=self.config.pipeline_name,
                execution_time=datetime.now(),
                duration_seconds=0,
                success=False,
                errors=[{"source": "pipeline", "message": error_message}]
            )
    
    def _log_info(self, message: str):
        """Log an info message."""
        log_entry = {
            "level": "INFO",
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.execution_log.append(log_entry)
        print(f"[INFO] Pipeline: {message}")
    
    def _log_warning(self, message: str):
        """Log a warning message."""
        log_entry = {
            "level": "WARNING",
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.execution_log.append(log_entry)
        print(f"[WARNING] Pipeline: {message}")
    
    def _log_error(self, message: str):
        """Log an error message."""
        log_entry = {
            "level": "ERROR",
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.execution_log.append(log_entry)
        print(f"[ERROR] Pipeline: {message}")
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the pipeline execution log."""
        return self.execution_log.copy()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about all agents."""
        return {
            agent_name: agent.get_agent_info()
            for agent_name, agent in self.agents.items()
        }