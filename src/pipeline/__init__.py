"""
SQL Generation Pipeline Package

This package implements a complex SQL generation pipeline that:
1. Processes external prompts
2. Generates prompt variants
3. Evaluates relevance with external descriptions
4. Generates SQL code
5. Merges multiple queries if needed
6. Translates to target SQL dialect
7. Outputs final SQL to file

Architecture:
- Chain of Responsibility: Each step handles specific task
- Strategy Pattern: Different processing strategies
- Factory Pattern: Component creation
- Observer Pattern: Pipeline monitoring
"""

__version__ = "1.0.0"
__author__ = "NOVASQLAgent"

from .main import SQLGenerationPipeline
from .models.pipeline_config import PipelineConfig
from .models.pipeline_state import PipelineState

__all__ = [
    "SQLGenerationPipeline",
    "PipelineConfig", 
    "PipelineState"
]