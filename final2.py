#!/usr/bin/env python3
"""
NOVA SQL Agent - Final2.py: Real Benchmark Data Processor

Sistema multi-agente per elaborare i dataset reali di Spider 2.0 e ELT-Bench.
Questo file lavora con i dati veri dei benchmark scaricati nella cartella 'data'.

Basato su final.py ma utilizza dataset reali invece di task di esempio.
"""

import os
import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

# Add paths for imports
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Core imports
try:
    from spider_agent.agent import NOVASQLSpiderAgent
    from spider_agent.config import SpiderAgentConfig, create_default_config
    from spider_agent.run import run_task
    
    # Model imports
    from src.model.LLMasJudge import LLMasJudgeSystem
    from src.model.LLMsql import SQLLLMGenerator, SQLGenerationConfig, SQLDialect
    from src.model.LLMmerge import LLMSQLMergerSystem
    from src.model.LLMrag import RAGAgent, RAGConfig
    from src.model.AgentHF import CodeAgent
    from src.model.LLMplanner import RequestPlanner, PlanningConfig
    from src.model.LLMsum import TextSummarizer, SummarizationConfig
    from src.model.LLMpromptopt import PromptOptimizer, PromptOptimizationConfig
    from src.SQLTranlator.sql_translator import SQLTranslator, SQLDialect as TranslatorDialect
    from src.vectorDB.vectorDB_manager import VectorDBManager
    
except ImportError as e:
    logging.warning(f"Some components not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkDataType(Enum):
    """Real benchmark data types."""
    SPIDER2_LITE = "spider2-lite"
    SPIDER2_SNOW = "spider2-snow"
    SPIDER2_DBT = "spider2-dbt"
    ELT_BENCH = "elt-bench"


@dataclass
class BenchmarkTask:
    """Real benchmark task structure."""
    task_id: str
    benchmark_type: BenchmarkDataType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result from processing real benchmark data."""
    task_id: str
    benchmark_type: BenchmarkDataType
    success: bool
    generated_sql: Optional[str] = None
    output_file: Optional[str] = None
    execution_time: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    benchmark_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealBenchmarkProcessor:
    """
    Processore per dataset reali di Spider 2.0 e ELT-Bench.
    
    Legge i file di benchmark reali dalla cartella 'data' e li elabora
    utilizzando il sistema multi-agente NOVASQLAgent.
    """
    
    def __init__(self, data_dir: str = "./data", output_dir: str = "./final2_results"):
        """Initialize processor with data directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.agents = {}
        self.results = []
        
        # Track processing statistics
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "processing_start": None,
            "processing_end": None
        }
        
        logger.info(f"🚀 RealBenchmarkProcessor initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📂 Output directory: {self.output_dir}")
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agent components for benchmark processing."""
        try:
            # Spider Agent for main processing
            spider_config = create_default_config()
            spider_config.model.name = "microsoft/phi-4-mini-instruct"
            self.agents['spider'] = NOVASQLSpiderAgent(spider_config)
            
            # LLM components
            self.agents['judge'] = LLMasJudgeSystem()
            
            sql_config = SQLGenerationConfig(
                model_name="microsoft/phi-4-mini-instruct",
                dialect=SQLDialect.POSTGRESQL
            )
            self.agents['sql_generator'] = SQLLLMGenerator(config=sql_config)
            self.agents['sql_merger'] = LLMSQLMergerSystem()
            self.agents['sql_translator'] = SQLTranslator()
            
            # Enhanced components
            rag_config = RAGConfig(model_name="microsoft/phi-4-mini-instruct")
            self.agents['rag'] = RAGAgent(config=rag_config)
            self.agents['code_agent'] = CodeAgent()
            
            planning_config = PlanningConfig(model_name="microsoft/phi-4-mini-instruct")
            self.agents['planner'] = RequestPlanner(config=planning_config)
            
            prompt_config = PromptOptimizationConfig(model_name="microsoft/phi-4-mini-instruct")
            self.agents['prompt_optimizer'] = PromptOptimizer(config=prompt_config)
            
            self.agents['vector_db'] = VectorDBManager()
            
            logger.info(f"✅ Initialized {len(self.agents)} agent components")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            raise
    
    def discover_benchmark_data(self) -> List[BenchmarkTask]:
        """
        Discover and load real benchmark datasets from data directory.
        
        Expected structure:
        data/
        ├── spider2-lite/
        │   ├── questions.jsonl
        │   └── databases/
        ├── spider2-snow/
        │   ├── snowflake_queries.json
        │   └── schemas/
        ├── spider2-dbt/
        │   ├── dbt_tasks.json
        │   └── models/
        └── elt-bench/
            ├── tasks.json
            └── data_streams/
        """
        tasks = []
        
        # Check if data directory exists
        if not self.data_dir.exists():
            logger.warning(f"📂 Data directory {self.data_dir} not found!")
            logger.info("🔧 Please download benchmark datasets first using:")
            logger.info("   python spider2_downloader.py")
            logger.info("   python elt_bench_downloader.py")
            return tasks
        
        try:
            # Spider 2.0 Lite dataset
            spider2_lite_dir = self.data_dir / "spider2-lite"
            if spider2_lite_dir.exists():
                tasks.extend(self._load_spider2_lite_data(spider2_lite_dir))
            
            # Spider 2.0 Snowflake dataset
            spider2_snow_dir = self.data_dir / "spider2-snow"
            if spider2_snow_dir.exists():
                tasks.extend(self._load_spider2_snow_data(spider2_snow_dir))
            
            # Spider 2.0 DBT dataset
            spider2_dbt_dir = self.data_dir / "spider2-dbt"
            if spider2_dbt_dir.exists():
                tasks.extend(self._load_spider2_dbt_data(spider2_dbt_dir))
            
            # ELT-Bench dataset
            elt_bench_dir = self.data_dir / "elt-bench"
            if elt_bench_dir.exists():
                tasks.extend(self._load_elt_bench_data(elt_bench_dir))
            
            logger.info(f"🔍 Discovered {len(tasks)} benchmark tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"❌ Error discovering benchmark data: {e}")
            return []
    
    def _load_spider2_lite_data(self, data_dir: Path) -> List[BenchmarkTask]:
        """Load Spider 2.0 Lite dataset."""
        tasks = []
        
        try:
            # Look for questions file (JSONL format)
            questions_file = data_dir / "questions.jsonl"
            if questions_file.exists():
                logger.info(f"📖 Loading Spider 2.0 Lite from {questions_file}")
                
                with open(questions_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            
                            # Extract Spider 2.0 format: instance_id, db, question, external_knowledge
                            task = BenchmarkTask(
                                task_id=f"spider2_lite_{data.get('instance_id', line_num)}",
                                benchmark_type=BenchmarkDataType.SPIDER2_LITE,
                                data={
                                    'instance_id': data.get('instance_id', f"lite_{line_num}"),
                                    'question': data.get('question', ''),
                                    'db': data.get('db', ''),
                                    'external_knowledge': data.get('external_knowledge', []),
                                    'difficulty': data.get('difficulty', 'unknown')
                                },
                                metadata={
                                    'source_file': str(questions_file),
                                    'line_number': line_num,
                                    'database': data.get('db', ''),
                                    'has_external_knowledge': bool(data.get('external_knowledge'))
                                }
                            )
                            tasks.append(task)
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Invalid JSON at line {line_num}: {e}")
                            continue
                            
                logger.info(f"✅ Loaded {len(tasks)} Spider 2.0 Lite tasks")
            else:
                logger.warning(f"📄 Questions file not found: {questions_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading Spider 2.0 Lite data: {e}")
            
        return tasks
    
    def _load_spider2_snow_data(self, data_dir: Path) -> List[BenchmarkTask]:
        """Load Spider 2.0 Snowflake dataset."""
        tasks = []
        
        try:
            # Look for Snowflake queries file
            queries_file = data_dir / "snowflake_queries.json"
            if queries_file.exists():
                logger.info(f"❄️ Loading Spider 2.0 Snowflake from {queries_file}")
                
                with open(queries_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    
                    for idx, data in enumerate(data_list, 1):
                        task = BenchmarkTask(
                            task_id=f"spider2_snow_{data.get('id', idx)}",
                            benchmark_type=BenchmarkDataType.SPIDER2_SNOW,
                            data={
                                'id': data.get('id', f"snow_{idx}"),
                                'question': data.get('question', ''),
                                'schema': data.get('schema', {}),
                                'snowflake_features': data.get('snowflake_features', []),
                                'complexity': data.get('complexity', 'medium')
                            },
                            metadata={
                                'source_file': str(queries_file),
                                'index': idx,
                                'uses_snowflake_features': bool(data.get('snowflake_features')),
                                'schema_tables': len(data.get('schema', {}).get('tables', []))
                            }
                        )
                        tasks.append(task)
                        
                logger.info(f"✅ Loaded {len(tasks)} Spider 2.0 Snowflake tasks")
            else:
                logger.warning(f"📄 Snowflake queries file not found: {queries_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading Spider 2.0 Snowflake data: {e}")
            
        return tasks
    
    def _load_spider2_dbt_data(self, data_dir: Path) -> List[BenchmarkTask]:
        """Load Spider 2.0 DBT dataset."""
        tasks = []
        
        try:
            # Look for DBT tasks file
            tasks_file = data_dir / "dbt_tasks.json"
            if tasks_file.exists():
                logger.info(f"🔧 Loading Spider 2.0 DBT from {tasks_file}")
                
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    
                    for idx, data in enumerate(data_list, 1):
                        task = BenchmarkTask(
                            task_id=f"spider2_dbt_{data.get('task_id', idx)}",
                            benchmark_type=BenchmarkDataType.SPIDER2_DBT,
                            data={
                                'task_id': data.get('task_id', f"dbt_{idx}"),
                                'instruction': data.get('instruction', ''),
                                'dbt_models': data.get('dbt_models', []),
                                'expected_output': data.get('expected_output', {}),
                                'difficulty': data.get('difficulty', 'intermediate')
                            },
                            metadata={
                                'source_file': str(tasks_file),
                                'index': idx,
                                'model_count': len(data.get('dbt_models', [])),
                                'has_expected_output': bool(data.get('expected_output'))
                            }
                        )
                        tasks.append(task)
                        
                logger.info(f"✅ Loaded {len(tasks)} Spider 2.0 DBT tasks")
            else:
                logger.warning(f"📄 DBT tasks file not found: {tasks_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading Spider 2.0 DBT data: {e}")
            
        return tasks
    
    def _load_elt_bench_data(self, data_dir: Path) -> List[BenchmarkTask]:
        """Load ELT-Bench dataset."""
        tasks = []
        
        try:
            # Look for ELT-Bench tasks file
            tasks_file = data_dir / "tasks.json"
            if tasks_file.exists():
                logger.info(f"🔄 Loading ELT-Bench from {tasks_file}")
                
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    
                    for idx, data in enumerate(data_list, 1):
                        task = BenchmarkTask(
                            task_id=f"elt_bench_{data.get('task_id', idx)}",
                            benchmark_type=BenchmarkDataType.ELT_BENCH,
                            data={
                                'task_id': data.get('task_id', f"elt_{idx}"),
                                'instruction': data.get('instruction', ''),
                                'data_sources': data.get('data_sources', []),
                                'transformations': data.get('transformations', []),
                                'target_schema': data.get('target_schema', {}),
                                'evaluation_criteria': data.get('evaluation_criteria', [])
                            },
                            metadata={
                                'source_file': str(tasks_file),
                                'index': idx,
                                'source_count': len(data.get('data_sources', [])),
                                'transformation_count': len(data.get('transformations', [])),
                                'has_target_schema': bool(data.get('target_schema'))
                            }
                        )
                        tasks.append(task)
                        
                logger.info(f"✅ Loaded {len(tasks)} ELT-Bench tasks")
            else:
                logger.warning(f"📄 ELT-Bench tasks file not found: {tasks_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading ELT-Bench data: {e}")
            
        return tasks
    
    async def process_spider2_task(self, task: BenchmarkTask) -> ProcessingResult:
        """Process Spider 2.0 benchmark task."""
        start_time = time.time()
        result = ProcessingResult(
            task_id=task.task_id,
            benchmark_type=task.benchmark_type,
            success=False
        )
        
        try:
            logger.info(f"🕷️ Processing Spider 2.0 task: {task.task_id}")
            
            # Extract task data based on benchmark type
            if task.benchmark_type == BenchmarkDataType.SPIDER2_LITE:
                question = task.data.get('question', '')
                external_knowledge = task.data.get('external_knowledge', [])
                database = task.data.get('db', '')
                
                # Prepare context descriptions
                descriptions = []
                if external_knowledge:
                    if isinstance(external_knowledge, list):
                        descriptions.extend([str(k) for k in external_knowledge])
                    else:
                        descriptions.append(str(external_knowledge))
                
                if database:
                    descriptions.append(f"Database: {database}")
                
                dialect = "postgresql"
                
            elif task.benchmark_type == BenchmarkDataType.SPIDER2_SNOW:
                question = task.data.get('question', '')
                schema = task.data.get('schema', {})
                snowflake_features = task.data.get('snowflake_features', [])
                
                # Prepare schema descriptions
                descriptions = []
                if schema and 'tables' in schema:
                    for table in schema['tables']:
                        table_desc = f"Table {table.get('name', 'unknown')}: {', '.join(table.get('columns', []))}"
                        descriptions.append(table_desc)
                
                if snowflake_features:
                    descriptions.append(f"Snowflake features: {', '.join(snowflake_features)}")
                
                dialect = "snowflake"
                
            elif task.benchmark_type == BenchmarkDataType.SPIDER2_DBT:
                question = task.data.get('instruction', '')
                dbt_models = task.data.get('dbt_models', [])
                
                # Prepare DBT model descriptions
                descriptions = []
                for model in dbt_models:
                    if isinstance(model, dict):
                        model_desc = f"DBT Model {model.get('name', 'unknown')}: {model.get('description', 'No description')}"
                        descriptions.append(model_desc)
                    else:
                        descriptions.append(str(model))
                
                dialect = "postgresql"  # Default for DBT
                
            else:
                question = str(task.data)
                descriptions = []
                dialect = "postgresql"
            
            # Use Spider Agent for processing
            spider_result = await asyncio.to_thread(
                run_task,
                prompt=question,
                descriptions=descriptions,
                dialect=dialect,
                config=None
            )
            
            result.agents_used.append('spider')
            
            if spider_result.get('success'):
                result.success = True
                result.generated_sql = spider_result.get('final_query', '')
                result.output_file = spider_result.get('output_file', '')
                
                # Calculate benchmark score (simplified)
                result.benchmark_score = 1.0 if result.generated_sql else 0.0
                
                # Save result to benchmark-specific output
                await self._save_benchmark_result(task, result, spider_result)
                
            else:
                result.errors.append(f"Spider Agent failed: {spider_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"❌ Error processing Spider 2.0 task {task.task_id}: {e}")
            result.errors.append(str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    async def process_elt_bench_task(self, task: BenchmarkTask) -> ProcessingResult:
        """Process ELT-Bench task."""
        start_time = time.time()
        result = ProcessingResult(
            task_id=task.task_id,
            benchmark_type=task.benchmark_type,
            success=False
        )
        
        try:
            logger.info(f"🔄 Processing ELT-Bench task: {task.task_id}")
            
            instruction = task.data.get('instruction', '')
            data_sources = task.data.get('data_sources', [])
            transformations = task.data.get('transformations', [])
            target_schema = task.data.get('target_schema', {})
            
            # Prepare ELT context
            descriptions = []
            
            # Add data source descriptions
            for source in data_sources:
                if isinstance(source, dict):
                    source_desc = f"Data source {source.get('name', 'unknown')}: {source.get('description', 'No description')}"
                    descriptions.append(source_desc)
                else:
                    descriptions.append(str(source))
            
            # Add transformation descriptions
            for transform in transformations:
                if isinstance(transform, dict):
                    transform_desc = f"Transformation: {transform.get('type', 'unknown')} - {transform.get('description', 'No description')}"
                    descriptions.append(transform_desc)
                else:
                    descriptions.append(str(transform))
            
            # Add target schema information
            if target_schema:
                schema_desc = f"Target schema: {json.dumps(target_schema, indent=2)}"
                descriptions.append(schema_desc)
            
            # Use Spider Agent for ELT processing
            spider_result = await asyncio.to_thread(
                run_task,
                prompt=instruction,
                descriptions=descriptions,
                dialect="postgresql",  # Default for ELT
                config=None
            )
            
            result.agents_used.append('spider')
            
            if spider_result.get('success'):
                result.success = True
                result.generated_sql = spider_result.get('final_query', '')
                result.output_file = spider_result.get('output_file', '')
                
                # Calculate ELT benchmark score (based on completeness)
                score_factors = [
                    1.0 if result.generated_sql else 0.0,
                    0.5 if data_sources else 0.0,
                    0.3 if transformations else 0.0,
                    0.2 if target_schema else 0.0
                ]
                result.benchmark_score = sum(score_factors) / len(score_factors)
                
                # Save result to benchmark-specific output
                await self._save_benchmark_result(task, result, spider_result)
                
            else:
                result.errors.append(f"ELT processing failed: {spider_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"❌ Error processing ELT-Bench task {task.task_id}: {e}")
            result.errors.append(str(e))
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _save_benchmark_result(self, task: BenchmarkTask, result: ProcessingResult, spider_result: Dict):
        """Save benchmark result in appropriate format."""
        try:
            # Create benchmark-specific output directory
            benchmark_output_dir = self.output_dir / task.benchmark_type.value
            benchmark_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed result
            result_file = benchmark_output_dir / f"{task.task_id}_result.json"
            result_data = {
                'task_id': task.task_id,
                'benchmark_type': task.benchmark_type.value,
                'original_data': task.data,
                'processing_result': {
                    'success': result.success,
                    'generated_sql': result.generated_sql,
                    'execution_time': result.execution_time,
                    'agents_used': result.agents_used,
                    'benchmark_score': result.benchmark_score,
                    'errors': result.errors
                },
                'spider_agent_result': spider_result,
                'metadata': {
                    **task.metadata,
                    'processed_at': datetime.now().isoformat(),
                    'output_file': result.output_file
                }
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
            
            # For Spider 2.0, also save in submission format
            if task.benchmark_type in [BenchmarkDataType.SPIDER2_LITE, BenchmarkDataType.SPIDER2_SNOW, BenchmarkDataType.SPIDER2_DBT]:
                await self._save_spider2_submission(task, result, benchmark_output_dir)
            
            # For ELT-Bench, save in evaluation format
            elif task.benchmark_type == BenchmarkDataType.ELT_BENCH:
                await self._save_elt_bench_submission(task, result, benchmark_output_dir)
            
            logger.info(f"💾 Benchmark result saved: {result_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save benchmark result: {e}")
    
    async def _save_spider2_submission(self, task: BenchmarkTask, result: ProcessingResult, output_dir: Path):
        """Save Spider 2.0 result in submission format."""
        try:
            # Create SQL file for submission
            sql_file = output_dir / f"{task.task_id}.sql"
            
            if result.generated_sql:
                # Add header for Spider 2.0 submission
                header = f"-- Spider 2.0 Submission\n-- Task ID: {task.task_id}\n-- Generated: {datetime.now().isoformat()}\n\n"
                
                with open(sql_file, 'w', encoding='utf-8') as f:
                    f.write(header + result.generated_sql)
                
                logger.info(f"📄 Spider 2.0 SQL saved: {sql_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save Spider 2.0 submission: {e}")
    
    async def _save_elt_bench_submission(self, task: BenchmarkTask, result: ProcessingResult, output_dir: Path):
        """Save ELT-Bench result in evaluation format."""
        try:
            # Create submission file for ELT-Bench
            submission_file = output_dir / f"{task.task_id}_submission.json"
            
            submission_data = {
                'task_id': task.task_id,
                'generated_solution': result.generated_sql,
                'execution_time': result.execution_time,
                'success': result.success,
                'score': result.benchmark_score,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(submission_file, 'w', encoding='utf-8') as f:
                json.dump(submission_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 ELT-Bench submission saved: {submission_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save ELT-Bench submission: {e}")
    
    async def process_all_tasks(self, tasks: List[BenchmarkTask], max_concurrent: int = 5) -> List[ProcessingResult]:
        """Process all benchmark tasks."""
        if not tasks:
            logger.warning("⚠️ No tasks to process!")
            return []
        
        logger.info(f"🚀 Processing {len(tasks)} benchmark tasks with max {max_concurrent} concurrent")
        
        self.stats["total_tasks"] = len(tasks)
        self.stats["processing_start"] = datetime.now()
        
        # Process with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(task: BenchmarkTask) -> ProcessingResult:
            async with semaphore:
                if task.benchmark_type == BenchmarkDataType.ELT_BENCH:
                    return await self.process_elt_bench_task(task)
                else:
                    return await self.process_spider2_task(task)
        
        # Process all tasks
        results = await asyncio.gather(*[
            process_with_semaphore(task) for task in tasks
        ], return_exceptions=True)
        
        # Handle exceptions and collect valid results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Task {tasks[i].task_id} failed with exception: {result}")
                error_result = ProcessingResult(
                    task_id=tasks[i].task_id,
                    benchmark_type=tasks[i].benchmark_type,
                    success=False,
                    errors=[str(result)]
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        # Update statistics
        self.stats["processing_end"] = datetime.now()
        self.stats["successful_tasks"] = sum(1 for r in valid_results if r.success)
        self.stats["failed_tasks"] = len(valid_results) - self.stats["successful_tasks"]
        
        self.results.extend(valid_results)
        
        # Generate summary report
        await self._generate_benchmark_report(valid_results)
        
        return valid_results
    
    async def _generate_benchmark_report(self, results: List[ProcessingResult]):
        """Generate comprehensive benchmark report."""
        try:
            report_file = self.output_dir / "benchmark_report.json"
            
            # Calculate statistics by benchmark type
            benchmark_stats = {}
            for result in results:
                benchmark_type = result.benchmark_type.value
                if benchmark_type not in benchmark_stats:
                    benchmark_stats[benchmark_type] = {
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "avg_score": 0.0,
                        "avg_execution_time": 0.0,
                        "scores": []
                    }
                
                stats = benchmark_stats[benchmark_type]
                stats["total"] += 1
                
                if result.success:
                    stats["successful"] += 1
                    if result.benchmark_score is not None:
                        stats["scores"].append(result.benchmark_score)
                else:
                    stats["failed"] += 1
            
            # Calculate averages
            for benchmark_type, stats in benchmark_stats.items():
                if stats["scores"]:
                    stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
                    stats["success_rate"] = (stats["successful"] / stats["total"]) * 100
                
                total_exec_time = sum(r.execution_time for r in results if r.benchmark_type.value == benchmark_type)
                stats["avg_execution_time"] = total_exec_time / stats["total"] if stats["total"] > 0 else 0.0
            
            # Generate final report
            report = {
                "summary": {
                    "total_tasks": self.stats["total_tasks"],
                    "successful_tasks": self.stats["successful_tasks"],
                    "failed_tasks": self.stats["failed_tasks"],
                    "overall_success_rate": (self.stats["successful_tasks"] / self.stats["total_tasks"]) * 100 if self.stats["total_tasks"] > 0 else 0,
                    "processing_start": self.stats["processing_start"].isoformat() if self.stats["processing_start"] else None,
                    "processing_end": self.stats["processing_end"].isoformat() if self.stats["processing_end"] else None,
                    "total_processing_time": (self.stats["processing_end"] - self.stats["processing_start"]).total_seconds() if self.stats["processing_start"] and self.stats["processing_end"] else 0
                },
                "benchmark_statistics": benchmark_stats,
                "error_analysis": self._analyze_errors(results),
                "generated_at": datetime.now().isoformat()
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 Benchmark report generated: {report_file}")
            
            # Print summary to console
            print("\n📊 " + "="*60)
            print("📊 BENCHMARK PROCESSING SUMMARY")
            print("📊 " + "="*60)
            print(f"✅ Total Tasks: {report['summary']['total_tasks']}")
            print(f"🎯 Successful: {report['summary']['successful_tasks']}")
            print(f"❌ Failed: {report['summary']['failed_tasks']}")
            print(f"📈 Success Rate: {report['summary']['overall_success_rate']:.1f}%")
            
            print(f"\n🏆 Benchmark Performance:")
            for benchmark_type, stats in benchmark_stats.items():
                print(f"   {benchmark_type}: {stats['successful']}/{stats['total']} ({stats.get('success_rate', 0):.1f}%) - Avg Score: {stats['avg_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate benchmark report: {e}")
    
    def _analyze_errors(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Analyze errors from processing results."""
        error_analysis = {
            "total_errors": 0,
            "error_types": {},
            "failed_tasks_by_benchmark": {},
            "common_errors": []
        }
        
        all_errors = []
        for result in results:
            if result.errors:
                error_analysis["total_errors"] += len(result.errors)
                all_errors.extend(result.errors)
                
                benchmark_type = result.benchmark_type.value
                if benchmark_type not in error_analysis["failed_tasks_by_benchmark"]:
                    error_analysis["failed_tasks_by_benchmark"][benchmark_type] = 0
                error_analysis["failed_tasks_by_benchmark"][benchmark_type] += 1
        
        # Find common error patterns
        from collections import Counter
        error_counter = Counter(all_errors)
        error_analysis["common_errors"] = error_counter.most_common(5)
        
        return error_analysis


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NOVA SQL Agent - Real Benchmark Data Processor for Spider 2.0 and ELT-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all benchmark data
  python final2.py --data-dir ./data --output-dir ./final2_results
  
  # Process only Spider 2.0 Lite
  python final2.py --benchmark spider2-lite
  
  # Process with limited concurrency
  python final2.py --max-concurrent 3
  
  # Verbose processing
  python final2.py --verbose --data-dir ./downloaded_data
        """
    )
    
    parser.add_argument('--data-dir', default='./data',
                       help='Directory containing benchmark datasets')
    parser.add_argument('--output-dir', default='./final2_results',
                       help='Output directory for results')
    
    parser.add_argument('--benchmark', choices=['spider2-lite', 'spider2-snow', 'spider2-dbt', 'elt-bench'],
                       help='Process only specific benchmark type')
    
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Maximum concurrent task processing')
    
    parser.add_argument('--limit', type=int,
                       help='Limit number of tasks to process (for testing)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


async def main():
    """Main function for processing real benchmark data."""
    args = parse_arguments()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    print("🚀 " + "="*60)
    print("🚀 NOVA SQL Agent - Real Benchmark Data Processor")
    print("🚀 " + "="*60)
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"📂 Output Directory: {args.output_dir}")
    print(f"🎯 Benchmark Filter: {args.benchmark or 'All benchmarks'}")
    print(f"⚡ Max Concurrent: {args.max_concurrent}")
    print(f"🔧 Task Limit: {args.limit or 'No limit'}")
    print("🚀 " + "="*60)
    
    try:
        # Initialize processor
        processor = RealBenchmarkProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Discover benchmark data
        print("🔍 Discovering benchmark datasets...")
        tasks = processor.discover_benchmark_data()
        
        if not tasks:
            print("❌ No benchmark tasks found!")
            print("💡 Please ensure benchmark datasets are downloaded:")
            print("   python spider2_downloader.py")
            print("   python elt_bench_downloader.py")
            return
        
        # Filter by benchmark type if specified
        if args.benchmark:
            target_benchmark = BenchmarkDataType(args.benchmark)
            tasks = [t for t in tasks if t.benchmark_type == target_benchmark]
            print(f"🎯 Filtered to {len(tasks)} {args.benchmark} tasks")
        
        # Apply task limit if specified
        if args.limit and args.limit < len(tasks):
            tasks = tasks[:args.limit]
            print(f"📊 Limited to first {args.limit} tasks")
        
        print(f"📦 Processing {len(tasks)} benchmark tasks...")
        
        # Process all tasks
        start_time = time.time()
        results = await processor.process_all_tasks(tasks, args.max_concurrent)
        total_time = time.time() - start_time
        
        # Final summary
        print("\n🎉 " + "="*60)
        print("🎉 BENCHMARK PROCESSING COMPLETE!")
        print("🎉 " + "="*60)
        print(f"⏱️  Total Processing Time: {total_time:.2f}s")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Detailed report: {args.output_dir}/benchmark_report.json")
        print("🎉 " + "="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())