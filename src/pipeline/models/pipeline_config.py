"""
Pipeline configuration data models with comprehensive settings.

This module defines the configuration structure for the SQL generation pipeline,
including settings for each step and integration options.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


class SQLDialect(Enum):
    """Supported SQL dialects for translation."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    GENERIC = "generic"


class ProcessingMode(Enum):
    """Pipeline processing modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    OPTIMIZED = "optimized"


@dataclass
class ModelConfig:
    """Configuration for LLM models used in pipeline."""
    model_id: str = "microsoft/phi-4-mini-instruct"
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
    device: Optional[str] = None


@dataclass
class ConfigurationStepConfig:
    """Configuration for configuration reading/writing step."""
    config_file_path: str = "./pipeline_config.json"
    auto_create: bool = True
    backup_existing: bool = True
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class PromptVariantConfig:
    """Configuration for prompt variant generation."""
    num_variants: int = 3
    max_variant_length: int = 500
    similarity_threshold: float = 0.8
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class RelevanceEvaluationConfig:
    """Configuration for relevance evaluation step."""
    confidence_threshold: float = 0.7
    require_unanimous: bool = False
    max_descriptions: int = 10
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class SQLGenerationConfig:
    """Configuration for SQL generation step."""
    include_comments: bool = True
    format_output: bool = True
    validate_syntax: bool = True
    max_query_length: int = 2000
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class QueryMergeConfig:
    """Configuration for query merging step."""
    auto_merge: bool = True
    merge_strategy: str = "union"  # "union", "join", "cte", "subquery"
    validate_merged: bool = True
    max_merged_queries: int = 5


@dataclass
class TranslationConfig:
    """Configuration for SQL translation step."""
    target_dialect: Optional[SQLDialect] = None
    preserve_comments: bool = True
    optimize_for_dialect: bool = True
    validate_translation: bool = True


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    output_file: Optional[str] = None
    output_directory: str = "./output"
    backup_existing: bool = True
    include_metadata: bool = True
    timestamp_files: bool = True


@dataclass
class MonitoringConfig:
    """Configuration for pipeline monitoring and logging."""
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    track_performance: bool = True
    save_intermediate_results: bool = False


@dataclass 
class PipelineConfig:
    """
    Master configuration for the SQL generation pipeline.
    
    This class contains all configuration settings for each step of the pipeline,
    allowing fine-grained control over the entire process.
    """
    
    # Step-specific configurations
    configuration_step: ConfigurationStepConfig = field(default_factory=ConfigurationStepConfig)
    prompt_variant: PromptVariantConfig = field(default_factory=PromptVariantConfig)
    relevance_evaluation: RelevanceEvaluationConfig = field(default_factory=RelevanceEvaluationConfig)
    sql_generation: SQLGenerationConfig = field(default_factory=SQLGenerationConfig)
    query_merge: QueryMergeConfig = field(default_factory=QueryMergeConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global pipeline settings
    processing_mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: int = 300
    
    # Additional metadata
    pipeline_name: str = "sql_generation_pipeline"
    pipeline_version: str = "1.0.0"
    created_by: str = "NOVASQLAgent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "configuration_step": self.configuration_step.__dict__,
            "prompt_variant": self.prompt_variant.__dict__,
            "relevance_evaluation": self.relevance_evaluation.__dict__,
            "sql_generation": self.sql_generation.__dict__,
            "query_merge": self.query_merge.__dict__,
            "translation": self.translation.__dict__,
            "output": self.output.__dict__,
            "monitoring": self.monitoring.__dict__,
            "processing_mode": self.processing_mode.value,
            "enable_error_recovery": self.enable_error_recovery,
            "max_retry_attempts": self.max_retry_attempts,
            "timeout_seconds": self.timeout_seconds,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "created_by": self.created_by
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(
            configuration_step=ConfigurationStepConfig(**config_dict.get("configuration_step", {})),
            prompt_variant=PromptVariantConfig(**config_dict.get("prompt_variant", {})),
            relevance_evaluation=RelevanceEvaluationConfig(**config_dict.get("relevance_evaluation", {})),
            sql_generation=SQLGenerationConfig(**config_dict.get("sql_generation", {})),
            query_merge=QueryMergeConfig(**config_dict.get("query_merge", {})),
            translation=TranslationConfig(**config_dict.get("translation", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            monitoring=MonitoringConfig(**config_dict.get("monitoring", {})),
            processing_mode=ProcessingMode(config_dict.get("processing_mode", "sequential")),
            enable_error_recovery=config_dict.get("enable_error_recovery", True),
            max_retry_attempts=config_dict.get("max_retry_attempts", 3),
            timeout_seconds=config_dict.get("timeout_seconds", 300),
            pipeline_name=config_dict.get("pipeline_name", "sql_generation_pipeline"),
            pipeline_version=config_dict.get("pipeline_version", "1.0.0"),
            created_by=config_dict.get("created_by", "NOVASQLAgent")
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.prompt_variant.num_variants < 1:
            issues.append("Number of prompt variants must be at least 1")
            
        if self.relevance_evaluation.confidence_threshold < 0 or self.relevance_evaluation.confidence_threshold > 1:
            issues.append("Confidence threshold must be between 0 and 1")
            
        if self.max_retry_attempts < 0:
            issues.append("Max retry attempts cannot be negative")
            
        if self.timeout_seconds <= 0:
            issues.append("Timeout must be positive")
            
        return issues