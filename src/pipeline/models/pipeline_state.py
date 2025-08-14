"""
Pipeline state management for tracking data flow through the SQL generation pipeline.

This module defines the state structure that tracks all data and progress
through each step of the pipeline, enabling monitoring and error recovery.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class StepStatus(Enum):
    """Status of individual pipeline steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """Overall pipeline status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result container for individual pipeline steps."""
    step_name: str
    status: StepStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, output_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Mark step as completed with output data."""
        self.status = StepStatus.COMPLETED
        self.end_time = datetime.now()
        self.output_data = output_data
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if metadata:
            self.metadata.update(metadata)
    
    def fail(self, error_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Mark step as failed with error message."""
        self.status = StepStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if metadata:
            self.metadata.update(metadata)


@dataclass
class ConfigurationStepState:
    """State for configuration step."""
    config_file_read: bool = False
    config_data: Optional[Dict[str, Any]] = None
    config_file_path: Optional[str] = None
    backup_created: bool = False
    auto_created: bool = False


@dataclass
class PromptVariantState:
    """State for prompt variant generation step."""
    original_prompt: str = ""
    variants: List[str] = field(default_factory=list)
    variant_similarity_scores: List[float] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelevanceEvaluationState:
    """State for relevance evaluation step."""
    external_descriptions: List[str] = field(default_factory=list)
    variant_relevance_matrix: List[List[str]] = field(default_factory=list)  # variants x descriptions -> yes/no
    relevant_descriptions: List[str] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    evaluation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SQLGenerationState:
    """State for SQL generation step."""
    input_prompt: str = ""
    relevant_context: List[str] = field(default_factory=list)
    generated_queries: List[str] = field(default_factory=list)
    primary_query: Optional[str] = None
    query_metadata: Dict[str, Any] = field(default_factory=dict)
    syntax_validation: Optional[bool] = None


@dataclass
class QueryMergeState:
    """State for query merging step."""
    input_queries: List[str] = field(default_factory=list)
    merge_strategy_used: Optional[str] = None
    merged_query: Optional[str] = None
    merge_successful: bool = False
    merge_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationState:
    """State for SQL translation step."""
    input_query: Optional[str] = None
    source_dialect: Optional[str] = None
    target_dialect: Optional[str] = None
    translated_query: Optional[str] = None
    translation_successful: bool = False
    translation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputState:
    """State for output generation step."""
    final_query: Optional[str] = None
    output_file_path: Optional[str] = None
    backup_created: bool = False
    metadata_written: bool = False
    output_successful: bool = False


@dataclass
class PipelineState:
    """
    Master state container for the entire SQL generation pipeline.
    
    This class tracks all data, progress, and results through each step
    of the pipeline, enabling monitoring, debugging, and error recovery.
    """
    
    # Pipeline metadata
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_name: str = "sql_generation_pipeline"
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: PipelineStatus = PipelineStatus.INITIALIZING
    
    # Input data
    external_prompt: str = ""
    external_descriptions: List[str] = field(default_factory=list)
    
    # Step states
    configuration: ConfigurationStepState = field(default_factory=ConfigurationStepState)
    prompt_variants: PromptVariantState = field(default_factory=PromptVariantState)
    relevance_evaluation: RelevanceEvaluationState = field(default_factory=RelevanceEvaluationState)
    sql_generation: SQLGenerationState = field(default_factory=SQLGenerationState)
    query_merge: QueryMergeState = field(default_factory=QueryMergeState)
    translation: TranslationState = field(default_factory=TranslationState)
    output: OutputState = field(default_factory=OutputState)
    
    # Step results tracking
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    
    # Error handling
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    total_duration_seconds: Optional[float] = None
    step_durations: Dict[str, float] = field(default_factory=dict)
    
    def start_step(self, step_name: str, input_data: Optional[Dict[str, Any]] = None) -> StepResult:
        """Start tracking a new pipeline step."""
        step_result = StepResult(
            step_name=step_name,
            status=StepStatus.IN_PROGRESS,
            start_time=datetime.now(),
            input_data=input_data
        )
        self.step_results[step_name] = step_result
        return step_result
    
    def complete_step(self, step_name: str, output_data: Dict[str, Any], 
                     metadata: Optional[Dict[str, Any]] = None):
        """Mark a step as completed with results."""
        if step_name in self.step_results:
            self.step_results[step_name].complete(output_data, metadata)
            self.step_durations[step_name] = self.step_results[step_name].duration_seconds or 0
    
    def fail_step(self, step_name: str, error_message: str, 
                  metadata: Optional[Dict[str, Any]] = None):
        """Mark a step as failed with error details."""
        if step_name in self.step_results:
            self.step_results[step_name].fail(error_message, metadata)
            self.step_durations[step_name] = self.step_results[step_name].duration_seconds or 0
        
        # Add to error tracking
        self.add_error(step_name, error_message, metadata)
    
    def add_error(self, source: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an error to the pipeline error log."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "message": message,
            "metadata": metadata or {}
        }
        self.errors.append(error_entry)
    
    def add_warning(self, source: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a warning to the pipeline warning log."""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "message": message,
            "metadata": metadata or {}
        }
        self.warnings.append(warning_entry)
    
    def complete_pipeline(self, status: PipelineStatus = PipelineStatus.COMPLETED):
        """Mark the entire pipeline as completed."""
        self.status = status
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def get_step_status(self, step_name: str) -> Optional[StepStatus]:
        """Get the status of a specific step."""
        if step_name in self.step_results:
            return self.step_results[step_name].status
        return None
    
    def get_step_output(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get the output data of a specific step."""
        if step_name in self.step_results:
            return self.step_results[step_name].output_data
        return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        completed_steps = sum(1 for result in self.step_results.values() 
                             if result.status == StepStatus.COMPLETED)
        failed_steps = sum(1 for result in self.step_results.values() 
                          if result.status == StepStatus.FAILED)
        
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.total_duration_seconds,
            "total_steps": len(self.step_results),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "final_output_available": bool(self.output.final_query),
            "output_file": self.output.output_file_path
        }
    
    def has_errors(self) -> bool:
        """Check if pipeline has any errors."""
        return len(self.errors) > 0 or any(
            result.status == StepStatus.FAILED 
            for result in self.step_results.values()
        )
    
    def is_completed(self) -> bool:
        """Check if pipeline completed successfully."""
        return (self.status == PipelineStatus.COMPLETED and 
                not self.has_errors() and 
                bool(self.output.final_query))