"""
Result models for pipeline execution results and responses.

This module defines the structure for returning results from the pipeline
execution, including success/failure status and detailed metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from .pipeline_state import PipelineState, PipelineStatus


@dataclass
class StepResult:
    """Result from a single pipeline step execution."""
    step_name: str
    success: bool
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Comprehensive result from pipeline execution.
    
    This class contains all results, metadata, and status information
    from a complete pipeline execution.
    """
    
    # Execution metadata
    pipeline_id: str
    pipeline_name: str
    execution_time: datetime
    duration_seconds: float
    success: bool
    
    # Results
    final_sql: Optional[str] = None
    output_file_path: Optional[str] = None
    
    # Step-by-step results
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    
    # Input data used
    original_prompt: str = ""
    external_descriptions: List[str] = field(default_factory=list)
    
    # Intermediate results
    prompt_variants: List[str] = field(default_factory=list)
    relevant_descriptions: List[str] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    merged_query: Optional[str] = None
    translated_query: Optional[str] = None
    
    # Configuration used
    configuration_used: Optional[Dict[str, Any]] = None
    
    # Error information
    errors: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    
    # Performance metrics
    step_durations: Dict[str, float] = field(default_factory=dict)
    tokens_used: Optional[int] = None
    api_calls_made: Optional[int] = None
    
    @classmethod
    def from_pipeline_state(cls, state: PipelineState) -> 'PipelineResult':
        """Create PipelineResult from PipelineState."""
        
        # Extract step results
        step_results = {}
        for step_name, step_result in state.step_results.items():
            step_results[step_name] = StepResult(
                step_name=step_name,
                success=step_result.status.value == "completed",
                output_data=step_result.output_data,
                error_message=step_result.error_message,
                duration_seconds=step_result.duration_seconds,
                metadata=step_result.metadata
            )
        
        # Determine final SQL output
        final_sql = None
        if state.output.final_query:
            final_sql = state.output.final_query
        elif state.translation.translated_query:
            final_sql = state.translation.translated_query
        elif state.query_merge.merged_query:
            final_sql = state.query_merge.merged_query
        elif state.sql_generation.primary_query:
            final_sql = state.sql_generation.primary_query
        
        return cls(
            pipeline_id=state.pipeline_id,
            pipeline_name=state.pipeline_name,
            execution_time=state.start_time,
            duration_seconds=state.total_duration_seconds or 0,
            success=state.status == PipelineStatus.COMPLETED and not state.has_errors(),
            final_sql=final_sql,
            output_file_path=state.output.output_file_path,
            step_results=step_results,
            original_prompt=state.external_prompt,
            external_descriptions=state.external_descriptions,
            prompt_variants=state.prompt_variants.variants,
            relevant_descriptions=state.relevance_evaluation.relevant_descriptions,
            generated_queries=state.sql_generation.generated_queries,
            merged_query=state.query_merge.merged_query,
            translated_query=state.translation.translated_query,
            errors=[{"source": err["source"], "message": err["message"]} for err in state.errors],
            warnings=[{"source": warn["source"], "message": warn["message"]} for warn in state.warnings],
            step_durations=state.step_durations
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the pipeline execution."""
        return {
            "pipeline_id": self.pipeline_id,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "final_sql_available": bool(self.final_sql),
            "output_file": self.output_file_path,
            "steps_completed": sum(1 for result in self.step_results.values() if result.success),
            "steps_failed": sum(1 for result in self.step_results.values() if not result.success),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "prompt_variants_generated": len(self.prompt_variants),
            "relevant_descriptions_found": len(self.relevant_descriptions),
            "queries_generated": len(self.generated_queries)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "total_duration": self.duration_seconds,
            "step_durations": self.step_durations,
            "longest_step": max(self.step_durations.items(), key=lambda x: x[1]) if self.step_durations else None,
            "average_step_duration": sum(self.step_durations.values()) / len(self.step_durations) if self.step_durations else 0,
            "tokens_used": self.tokens_used,
            "api_calls_made": self.api_calls_made
        }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get a chronological log of pipeline execution."""
        log_entries = []
        
        # Add step results to log
        for step_name, result in self.step_results.items():
            log_entries.append({
                "type": "step_result",
                "step_name": step_name,
                "success": result.success,
                "duration": result.duration_seconds,
                "error": result.error_message
            })
        
        # Add errors to log
        for error in self.errors:
            log_entries.append({
                "type": "error",
                "source": error["source"],
                "message": error["message"]
            })
        
        # Add warnings to log
        for warning in self.warnings:
            log_entries.append({
                "type": "warning", 
                "source": warning["source"],
                "message": warning["message"]
            })
        
        return log_entries
    
    def export_results(self) -> Dict[str, Any]:
        """Export complete results in dictionary format."""
        return {
            "metadata": {
                "pipeline_id": self.pipeline_id,
                "pipeline_name": self.pipeline_name,
                "execution_time": self.execution_time.isoformat(),
                "duration_seconds": self.duration_seconds,
                "success": self.success
            },
            "input": {
                "original_prompt": self.original_prompt,
                "external_descriptions": self.external_descriptions
            },
            "outputs": {
                "final_sql": self.final_sql,
                "output_file_path": self.output_file_path,
                "prompt_variants": self.prompt_variants,
                "relevant_descriptions": self.relevant_descriptions,
                "generated_queries": self.generated_queries,
                "merged_query": self.merged_query,
                "translated_query": self.translated_query
            },
            "execution": {
                "step_results": {name: {
                    "success": result.success,
                    "duration_seconds": result.duration_seconds,
                    "error_message": result.error_message
                } for name, result in self.step_results.items()},
                "errors": self.errors,
                "warnings": self.warnings,
                "performance": self.get_performance_report()
            }
        }