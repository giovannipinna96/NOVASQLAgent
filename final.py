#!/usr/bin/env python3
"""
NOVA SQL Agent - Final Multi-Agent System Coordinator

Sistema multi-agente unificato per Spider 2.0 e ELT-Bench benchmarks.
Coordina tutti i componenti esistenti per risolvere task complessi di Text-to-SQL e ELT pipeline.
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
from spider_agent.agent import NOVASQLSpiderAgent
from spider_agent.config import SpiderAgentConfig, create_default_config
from spider_agent.run import run_task

# Model imports
try:
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


class BenchmarkType(Enum):
    """Supported benchmark types."""
    SPIDER2_LITE = "spider2-lite"
    SPIDER2_SNOW = "spider2-snow" 
    SPIDER2_DBT = "spider2-dbt"
    ELT_BENCH = "elt-bench"
    AUTO_DETECT = "auto"


class TaskType(Enum):
    """Task types supported by the system."""
    TEXT_TO_SQL = "text-to-sql"
    ELT_PIPELINE = "elt-pipeline"
    DATA_TRANSFORMATION = "data-transformation"
    MULTI_MODAL = "multi-modal"


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent system."""
    benchmark_type: BenchmarkType = BenchmarkType.AUTO_DETECT
    primary_model: str = "gpt-4o"
    fallback_model: str = "microsoft/phi-4-mini-instruct"
    max_agents: int = 10
    timeout_per_task: int = 300
    enable_rag: bool = True
    enable_planning: bool = True
    enable_prompt_optimization: bool = True
    enable_parallel_execution: bool = True
    output_dir: str = "./final_results"
    verbose: bool = True
    debug: bool = False


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    benchmark_type: BenchmarkType
    task_type: TaskType
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NOVAMultiAgentOrchestrator:
    """
    Orchestratore principale del sistema multi-agente NOVA SQL Agent.
    Coordina tutti i componenti per risolvere benchmark Spider 2.0 e ELT-Bench.
    """
    
    def __init__(self, config: MultiAgentConfig):
        """Initialize the multi-agent orchestrator."""
        self.config = config
        self.agents = {}
        self.results = []
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_agents()
        
        logger.info(f"🚀 NOVA Multi-Agent System initialized for {config.benchmark_type.value}")
    
    def _initialize_agents(self):
        """Initialize all agent components."""
        try:
            # Spider Agent (ELT-Bench compatible)
            spider_config = create_default_config()
            spider_config.model.name = self.config.primary_model
            spider_config.execution.verbose = self.config.verbose
            spider_config.execution.debug = self.config.debug
            self.agents['spider'] = NOVASQLSpiderAgent(spider_config)
            
            # LLM Judge System
            self.agents['judge'] = LLMasJudgeSystem()
            
            # SQL Generator
            sql_config = SQLGenerationConfig(
                model_name=self.config.primary_model,
                dialect=SQLDialect.POSTGRESQL
            )
            self.agents['sql_generator'] = SQLLLMGenerator(config=sql_config)
            
            # SQL Merger
            self.agents['sql_merger'] = LLMSQLMergerSystem()
            
            # SQL Translator
            self.agents['sql_translator'] = SQLTranslator()
            
            # RAG Agent (if enabled)
            if self.config.enable_rag:
                rag_config = RAGConfig(model_name=self.config.primary_model)
                self.agents['rag'] = RAGAgent(config=rag_config)
            
            # Code Agent
            self.agents['code_agent'] = CodeAgent()
            
            # Request Planner (if enabled)
            if self.config.enable_planning:
                planning_config = PlanningConfig(model_name=self.config.primary_model)
                self.agents['planner'] = RequestPlanner(config=planning_config)
            
            # Text Summarizer
            summarization_config = SummarizationConfig(model_name=self.config.primary_model)
            self.agents['summarizer'] = TextSummarizer(config=summarization_config)
            
            # Prompt Optimizer (if enabled)
            if self.config.enable_prompt_optimization:
                prompt_config = PromptOptimizationConfig(model_name=self.config.primary_model)
                self.agents['prompt_optimizer'] = PromptOptimizer(config=prompt_config)
            
            # Vector DB Manager
            self.agents['vector_db'] = VectorDBManager()
            
            logger.info(f"✅ Initialized {len(self.agents)} agent components")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
            raise
    
    def detect_benchmark_type(self, task_input: Dict[str, Any]) -> BenchmarkType:
        """Auto-detect benchmark type from task input."""
        if 'sql' in str(task_input).lower() or 'query' in str(task_input).lower():
            if 'bigquery' in str(task_input).lower():
                return BenchmarkType.SPIDER2_LITE
            elif 'snowflake' in str(task_input).lower():
                return BenchmarkType.SPIDER2_SNOW
            elif 'dbt' in str(task_input).lower():
                return BenchmarkType.SPIDER2_DBT
            else:
                return BenchmarkType.SPIDER2_LITE
        elif any(keyword in str(task_input).lower() for keyword in ['elt', 'etl', 'pipeline', 'transform']):
            return BenchmarkType.ELT_BENCH
        else:
            return BenchmarkType.SPIDER2_LITE  # Default
    
    def detect_task_type(self, task_input: Dict[str, Any]) -> TaskType:
        """Detect task type from input."""
        input_str = str(task_input).lower()
        
        if 'sql' in input_str or 'query' in input_str:
            return TaskType.TEXT_TO_SQL
        elif any(keyword in input_str for keyword in ['elt', 'etl', 'pipeline']):
            return TaskType.ELT_PIPELINE
        elif 'transform' in input_str or 'process' in input_str:
            return TaskType.DATA_TRANSFORMATION
        else:
            return TaskType.TEXT_TO_SQL  # Default
    
    async def process_spider2_task(self, task_input: Dict[str, Any], task_id: str) -> TaskResult:
        """Process Spider 2.0 benchmark task."""
        start_time = time.time()
        result = TaskResult(
            task_id=task_id,
            benchmark_type=self.config.benchmark_type,
            task_type=TaskType.TEXT_TO_SQL
        )
        
        try:
            logger.info(f"🕷️ Processing Spider 2.0 task: {task_id}")
            
            # Extract task components
            prompt = task_input.get('prompt', task_input.get('question', ''))
            descriptions = task_input.get('descriptions', task_input.get('context', []))
            dialect = task_input.get('dialect', 'postgresql')
            
            if not prompt:
                raise ValueError("No prompt/question found in task input")
            
            # Step 1: Optimize prompt if enabled
            if self.config.enable_prompt_optimization and 'prompt_optimizer' in self.agents:
                logger.info("🔧 Optimizing prompt...")
                optimization_result = self.agents['prompt_optimizer'].optimize_prompt(prompt)
                if optimization_result.success:
                    prompt = optimization_result.optimized_prompt
                    result.agents_used.append('prompt_optimizer')
            
            # Step 2: Plan request if enabled
            plan_steps = []
            if self.config.enable_planning and 'planner' in self.agents:
                logger.info("📋 Planning request...")
                planning_result = self.agents['planner'].plan_request(
                    prompt, 
                    strategy="sequential"
                )
                if planning_result.get('success'):
                    plan_steps = planning_result.get('steps', [])
                    result.agents_used.append('planner')
            
            # Step 3: Use Spider Agent for main processing
            logger.info("🕷️ Using Spider Agent for SQL generation...")
            spider_result = await asyncio.to_thread(
                run_task,
                prompt=prompt,
                descriptions=descriptions if isinstance(descriptions, list) else [descriptions],
                dialect=dialect,
                config=None  # Use default config
            )
            
            result.agents_used.append('spider')
            
            if spider_result.get('success'):
                result.success = True
                result.result_data = {
                    'sql_query': spider_result.get('final_query', ''),
                    'output_file': spider_result.get('output_file', ''),
                    'steps_completed': spider_result.get('steps_completed', ''),
                    'execution_time': spider_result.get('execution_time', 0),
                    'plan_steps': plan_steps
                }
            else:
                result.errors.append(f"Spider Agent failed: {spider_result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"❌ Error processing Spider 2.0 task {task_id}: {e}")
            result.errors.append(str(e))
            result.success = False
        
        result.execution_time = time.time() - start_time
        return result
    
    async def process_elt_bench_task(self, task_input: Dict[str, Any], task_id: str) -> TaskResult:
        """Process ELT-Bench task."""
        start_time = time.time()
        result = TaskResult(
            task_id=task_id,
            benchmark_type=BenchmarkType.ELT_BENCH,
            task_type=TaskType.ELT_PIPELINE
        )
        
        try:
            logger.info(f"🔄 Processing ELT-Bench task: {task_id}")
            
            # Extract task components
            instruction = task_input.get('instruction', task_input.get('prompt', ''))
            context = task_input.get('context', task_input.get('descriptions', []))
            
            if not instruction:
                raise ValueError("No instruction found in task input")
            
            # Step 1: Plan ELT pipeline if enabled
            pipeline_steps = []
            if self.config.enable_planning and 'planner' in self.agents:
                logger.info("📋 Planning ELT pipeline...")
                planning_result = self.agents['planner'].plan_request(
                    instruction,
                    strategy="dependency_based"
                )
                if planning_result.get('success'):
                    pipeline_steps = planning_result.get('steps', [])
                    result.agents_used.append('planner')
            
            # Step 2: Use RAG for context enhancement if enabled
            enhanced_context = context
            if self.config.enable_rag and 'rag' in self.agents:
                logger.info("🔍 Enhancing context with RAG...")
                try:
                    rag_result = self.agents['rag'].search_and_generate(
                        query=instruction,
                        context=str(context)
                    )
                    if rag_result.get('success'):
                        enhanced_context = rag_result.get('response', context)
                        result.agents_used.append('rag')
                except Exception as e:
                    logger.warning(f"RAG enhancement failed: {e}")
            
            # Step 3: Use Spider Agent for ELT execution
            logger.info("🕷️ Using Spider Agent for ELT execution...")
            spider_result = await asyncio.to_thread(
                run_task,
                prompt=instruction,
                descriptions=[str(enhanced_context)] if enhanced_context else [],
                dialect="postgresql",  # Default for ELT
                config=None
            )
            
            result.agents_used.append('spider')
            
            # Step 4: Execute any bash commands if needed
            if 'code_agent' in self.agents and pipeline_steps:
                logger.info("💻 Executing pipeline steps with Code Agent...")
                try:
                    for step in pipeline_steps[:3]:  # Limit to first 3 steps
                        step_instruction = step.get('instruction', step.get('content', ''))
                        if step_instruction:
                            code_result = self.agents['code_agent'].execute_request(step_instruction)
                            if code_result.get('success'):
                                result.agents_used.append('code_agent')
                except Exception as e:
                    logger.warning(f"Code execution failed: {e}")
            
            if spider_result.get('success'):
                result.success = True
                result.result_data = {
                    'pipeline_result': spider_result.get('final_query', ''),
                    'output_file': spider_result.get('output_file', ''),
                    'pipeline_steps': pipeline_steps,
                    'enhanced_context': enhanced_context != context,
                    'execution_time': spider_result.get('execution_time', 0)
                }
            else:
                result.errors.append(f"ELT execution failed: {spider_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"❌ Error processing ELT-Bench task {task_id}: {e}")
            result.errors.append(str(e))
            result.success = False
        
        result.execution_time = time.time() - start_time
        return result
    
    async def process_task(self, task_input: Dict[str, Any], task_id: Optional[str] = None) -> TaskResult:
        """Process a single task using appropriate benchmark handler."""
        if task_id is None:
            task_id = f"task_{len(self.results) + 1}_{int(time.time())}"
        
        # Auto-detect benchmark type if not specified
        benchmark_type = self.config.benchmark_type
        if benchmark_type == BenchmarkType.AUTO_DETECT:
            benchmark_type = self.detect_benchmark_type(task_input)
            logger.info(f"🔍 Auto-detected benchmark type: {benchmark_type.value}")
        
        # Route to appropriate processor
        if benchmark_type in [BenchmarkType.SPIDER2_LITE, BenchmarkType.SPIDER2_SNOW, BenchmarkType.SPIDER2_DBT]:
            result = await self.process_spider2_task(task_input, task_id)
        elif benchmark_type == BenchmarkType.ELT_BENCH:
            result = await self.process_elt_bench_task(task_input, task_id)
        else:
            # Default to Spider 2.0
            result = await self.process_spider2_task(task_input, task_id)
        
        # Store result
        self.results.append(result)
        
        # Save result to file
        self._save_result(result)
        
        return result
    
    async def process_batch(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Process multiple tasks in batch."""
        logger.info(f"📦 Processing batch of {len(tasks)} tasks...")
        
        if self.config.enable_parallel_execution and len(tasks) > 1:
            # Process tasks in parallel
            semaphore = asyncio.Semaphore(self.config.max_agents)
            
            async def process_with_semaphore(task, idx):
                async with semaphore:
                    return await self.process_task(task, f"batch_task_{idx}")
            
            results = await asyncio.gather(*[
                process_with_semaphore(task, idx) 
                for idx, task in enumerate(tasks)
            ], return_exceptions=True)
            
            # Handle exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                    error_result = TaskResult(
                        task_id=f"batch_task_{i}",
                        benchmark_type=BenchmarkType.AUTO_DETECT,
                        task_type=TaskType.TEXT_TO_SQL,
                        success=False,
                        errors=[str(result)]
                    )
                    valid_results.append(error_result)
                else:
                    valid_results.append(result)
            
            return valid_results
        else:
            # Process tasks sequentially
            results = []
            for idx, task in enumerate(tasks):
                result = await self.process_task(task, f"batch_task_{idx}")
                results.append(result)
            
            return results
    
    def _save_result(self, result: TaskResult):
        """Save individual result to file."""
        try:
            result_file = Path(self.config.output_dir) / f"{result.task_id}_result.json"
            
            result_data = {
                'task_id': result.task_id,
                'benchmark_type': result.benchmark_type.value,
                'task_type': result.task_type.value,
                'success': result.success,
                'execution_time': result.execution_time,
                'agents_used': result.agents_used,
                'result_data': result.result_data,
                'errors': result.errors,
                'metadata': result.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
            
            if self.config.verbose:
                logger.info(f"💾 Result saved: {result_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save result: {e}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all processed tasks."""
        if not self.results:
            return {"message": "No tasks processed yet"}
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tasks
        
        # Benchmark distribution
        benchmark_counts = {}
        for result in self.results:
            benchmark_type = result.benchmark_type.value
            benchmark_counts[benchmark_type] = benchmark_counts.get(benchmark_type, 0) + 1
        
        # Agent usage statistics
        agent_usage = {}
        for result in self.results:
            for agent in result.agents_used:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        report = {
            'summary': {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
                'avg_execution_time': avg_execution_time
            },
            'benchmark_distribution': benchmark_counts,
            'agent_usage_statistics': agent_usage,
            'errors': [error for result in self.results for error in result.errors],
            'generated_at': datetime.now().isoformat()
        }
        
        # Save summary report
        try:
            report_file = Path(self.config.output_dir) / "summary_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📊 Summary report saved: {report_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save summary report: {e}")
        
        return report


def create_example_tasks() -> List[Dict[str, Any]]:
    """Create example tasks for testing."""
    return [
        # Spider 2.0 tasks
        {
            "prompt": "Find all customers who made purchases in the last 30 days and their total spending",
            "descriptions": [
                "customers table: customer_id, name, email, registration_date",
                "orders table: order_id, customer_id, order_date, total_amount, status",
                "order_items table: item_id, order_id, product_id, quantity, unit_price"
            ],
            "dialect": "postgresql"
        },
        {
            "prompt": "Create a monthly sales report showing revenue trends by product category",
            "descriptions": [
                "sales table: sale_id, product_id, customer_id, sale_date, quantity, unit_price",
                "products table: product_id, product_name, category_id, price",
                "categories table: category_id, category_name, description"
            ],
            "dialect": "bigquery"
        },
        
        # ELT-Bench tasks
        {
            "instruction": "Create an ELT pipeline to extract user data from multiple sources, transform it for analytics, and load it into a data warehouse",
            "context": [
                "Source systems: CRM database (users, contacts), E-commerce platform (transactions, products)",
                "Target: Analytics data warehouse with star schema",
                "Requirements: Data quality checks, incremental updates, error handling"
            ]
        },
        {
            "instruction": "Build a real-time data processing pipeline for IoT sensor data with streaming analytics",
            "context": [
                "Data source: IoT sensors sending JSON messages via Kafka",
                "Processing: Aggregations, anomaly detection, alerting",
                "Output: Dashboard metrics, alerts, historical storage"
            ]
        }
    ]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NOVA SQL Agent - Multi-Agent System for Spider 2.0 and ELT-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with example tasks
  python final.py --example-tasks
  
  # Run with specific benchmark
  python final.py --benchmark spider2-lite --model gpt-4o
  
  # Process tasks from file
  python final.py --input-file tasks.json --output-dir ./results
  
  # Batch mode with parallel execution
  python final.py --input-file batch_tasks.json --parallel --max-agents 5
        """
    )
    
    parser.add_argument('--benchmark', choices=[b.value for b in BenchmarkType], 
                       default='auto', help='Benchmark type to use')
    parser.add_argument('--model', default='gpt-4o', 
                       help='Primary model to use')
    parser.add_argument('--fallback-model', default='microsoft/phi-4-mini-instruct',
                       help='Fallback model to use')
    
    parser.add_argument('--input-file', type=str,
                       help='JSON file containing tasks to process')
    parser.add_argument('--output-dir', default='./final_results',
                       help='Output directory for results')
    
    parser.add_argument('--example-tasks', action='store_true',
                       help='Run with example tasks')
    
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel task execution')
    parser.add_argument('--max-agents', type=int, default=5,
                       help='Maximum number of concurrent agents')
    
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per task in seconds')
    
    parser.add_argument('--disable-rag', action='store_true',
                       help='Disable RAG functionality')
    parser.add_argument('--disable-planning', action='store_true',
                       help='Disable request planning')
    parser.add_argument('--disable-prompt-opt', action='store_true',
                       help='Disable prompt optimization')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create configuration
    config = MultiAgentConfig(
        benchmark_type=BenchmarkType(args.benchmark),
        primary_model=args.model,
        fallback_model=args.fallback_model,
        max_agents=args.max_agents,
        timeout_per_task=args.timeout,
        enable_rag=not args.disable_rag,
        enable_planning=not args.disable_planning,
        enable_prompt_optimization=not args.disable_prompt_opt,
        enable_parallel_execution=args.parallel,
        output_dir=args.output_dir,
        verbose=args.verbose,
        debug=args.debug
    )
    
    print("🚀 " + "="*60)
    print("🚀 NOVA SQL Agent - Multi-Agent System Coordinator")
    print("🚀 " + "="*60)
    print(f"🎯 Benchmark: {config.benchmark_type.value}")
    print(f"🤖 Primary Model: {config.primary_model}")
    print(f"📁 Output Directory: {config.output_dir}")
    print(f"⚙️  Max Agents: {config.max_agents}")
    print(f"🔧 RAG: {'✅' if config.enable_rag else '❌'}")
    print(f"📋 Planning: {'✅' if config.enable_planning else '❌'}")
    print(f"🔍 Prompt Optimization: {'✅' if config.enable_prompt_optimization else '❌'}")
    print(f"⚡ Parallel Execution: {'✅' if config.enable_parallel_execution else '❌'}")
    print("🚀 " + "="*60)
    
    try:
        # Initialize orchestrator
        orchestrator = NOVAMultiAgentOrchestrator(config)
        
        # Load tasks
        tasks = []
        if args.input_file:
            print(f"📂 Loading tasks from: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    tasks = data
                elif isinstance(data, dict) and 'tasks' in data:
                    tasks = data['tasks']
                else:
                    tasks = [data]  # Single task
        elif args.example_tasks:
            print("📝 Using example tasks")
            tasks = create_example_tasks()
        else:
            print("❌ No tasks specified. Use --input-file or --example-tasks")
            return
        
        print(f"📦 Processing {len(tasks)} tasks...")
        
        # Process tasks
        start_time = time.time()
        results = await orchestrator.process_batch(tasks)
        total_time = time.time() - start_time
        
        # Generate summary
        print("\n📊 " + "="*60)
        print("📊 EXECUTION SUMMARY")
        print("📊 " + "="*60)
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"✅ Successful Tasks: {successful}/{len(results)}")
        print(f"❌ Failed Tasks: {failed}/{len(results)}")
        print(f"⏱️  Total Execution Time: {total_time:.2f}s")
        print(f"⚡ Average Time per Task: {total_time/len(results):.2f}s")
        
        # Agent usage statistics
        agent_usage = {}
        for result in results:
            for agent in result.agents_used:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        if agent_usage:
            print(f"\n🤖 Agent Usage:")
            for agent, count in sorted(agent_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"   {agent}: {count} times")
        
        # Show errors if any
        all_errors = [error for result in results for error in result.errors]
        if all_errors:
            print(f"\n❌ Errors encountered:")
            for error in all_errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(all_errors) > 5:
                print(f"   ... and {len(all_errors) - 5} more errors")
        
        # Generate detailed report
        report = orchestrator.generate_summary_report()
        
        print(f"\n💾 Results saved to: {config.output_dir}")
        print(f"📊 Summary report: {Path(config.output_dir) / 'summary_report.json'}")
        
        print("\n🎉 " + "="*60)
        print("🎉 NOVA Multi-Agent System Execution Complete!")
        print("🎉 " + "="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())