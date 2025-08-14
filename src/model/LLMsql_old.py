"""
LLM per generazione SQL da richieste in linguaggio naturale.

Sistema semplificato che usa HuggingFace Transformers per convertire 
richieste in linguaggio naturale in query SQL.

FUNZIONALIT� PRINCIPALI:
1. Generazione SQL da testo naturale
2. Support per diversi tipi di query (SELECT, INSERT, UPDATE, DELETE)
3. Template-based prompt engineering
4. Generazione con phi-4-mini-instruct per consistenza architetturale

ARCHITETTURA:
- SQLLLMGenerator: Core LLM per text-to-SQL usando pipeline Transformers
- Prompt engineering ottimizzato per generazione SQL
- Supporto per schemi database opzionali

UTILIZZO:
python LLMsql.py --model "microsoft/phi-4-mini-instruct" --query "Find all users with age > 25"
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import per HuggingFace Transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    GenerationConfig
)
import torch

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLDialect(Enum):
    """Dialetti SQL supportati per generazione."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    GENERIC = "generic"


class QueryType(Enum):
    """Tipi di query SQL."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE_TABLE = "create_table"
    ALTER_TABLE = "alter_table"
    DROP_TABLE = "drop_table"
    CREATE_INDEX = "create_index"
    COMPLEX = "complex"


@dataclass
class SQLGenerationConfig:
    """Configurazione per la generazione SQL."""
    model_name: str = "microsoft/phi-4-mini-instruct"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    return_full_text: bool = False


@dataclass
class SchemaInfo:
    """Informazioni dello schema database."""
    tables: Dict[str, List[str]]  # nome_tabella -> lista_colonne
    relationships: Dict[str, List[str]] = None  # tabella -> foreign_keys
    indexes: Dict[str, List[str]] = None  # tabella -> lista_indici
    constraints: Dict[str, List[str]] = None  # tabella -> vincoli


class SQLPromptBuilder:
    """Costruisce prompt strutturati per la generazione SQL."""
    
    def __init__(self, dialect: SQLDialect = SQLDialect.POSTGRESQL):
        self.dialect = dialect
        
        # Template di base per diversi tipi di query
        self.base_templates = {
            QueryType.SELECT: self._get_select_template(),
            QueryType.INSERT: self._get_insert_template(),
            QueryType.UPDATE: self._get_update_template(),
            QueryType.DELETE: self._get_delete_template(),
            QueryType.CREATE_TABLE: self._get_create_table_template(),
            QueryType.COMPLEX: self._get_complex_template()
        }
        
        # Few-shot examples per dialetto
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _get_select_template(self) -> str:
        return """-- Task: Generate a SQL SELECT query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: SELECT

-- Examples:
{examples}

-- Generated SQL Query:
SELECT"""
    
    def _get_insert_template(self) -> str:
        return """-- Task: Generate a SQL INSERT query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: INSERT

-- Examples:
{examples}

-- Generated SQL Query:
INSERT"""
    
    def _get_update_template(self) -> str:
        return """-- Task: Generate a SQL UPDATE query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: UPDATE

-- Examples:
{examples}

-- Generated SQL Query:
UPDATE"""
    
    def _get_delete_template(self) -> str:
        return """-- Task: Generate a SQL DELETE query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: DELETE

-- Examples:
{examples}

-- Generated SQL Query:
DELETE"""
    
    def _get_create_table_template(self) -> str:
        return """-- Task: Generate a SQL CREATE TABLE query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: CREATE TABLE

-- Examples:
{examples}

-- Generated SQL Query:
CREATE TABLE"""
    
    def _get_complex_template(self) -> str:
        return """-- Task: Generate a complex SQL query
-- Database: {dialect}
-- Schema: {schema_info}
-- Request: {user_request}
-- Query Type: COMPLEX (JOINs, Subqueries, CTEs, etc.)

-- Examples:
{examples}

-- Generated SQL Query:"""
    
    def _load_few_shot_examples(self) -> Dict[QueryType, List[str]]:
        """Carica esempi few-shot per tipo di query."""
        return {
            QueryType.SELECT: [
                "-- Find all users with age greater than 25\nSELECT * FROM users WHERE age > 25;",
                "-- Get total sales by product category\nSELECT category, SUM(amount) FROM sales GROUP BY category;",
                "-- Find top 10 customers by total orders\nSELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 10;"
            ],
            QueryType.INSERT: [
                "-- Insert a new user\nINSERT INTO users (name, email, age) VALUES ('John Doe', 'john@example.com', 30);",
                "-- Insert multiple products\nINSERT INTO products (name, price, category) VALUES ('Laptop', 999.99, 'Electronics'), ('Mouse', 29.99, 'Electronics');"
            ],
            QueryType.UPDATE: [
                "-- Update user email\nUPDATE users SET email = 'newemail@example.com' WHERE user_id = 1;",
                "-- Increase all product prices by 10%\nUPDATE products SET price = price * 1.10 WHERE category = 'Electronics';"
            ],
            QueryType.DELETE: [
                "-- Delete inactive users\nDELETE FROM users WHERE last_login < '2023-01-01';",
                "-- Delete orders older than 2 years\nDELETE FROM orders WHERE order_date < CURRENT_DATE - INTERVAL '2 years';"
            ],
            QueryType.CREATE_TABLE: [
                "-- Create users table\nCREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100), email VARCHAR(100) UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);",
                "-- Create orders table with foreign key\nCREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), total DECIMAL(10,2), order_date DATE);"
            ],
            QueryType.COMPLEX: [
                "-- Find customers with orders above average\nWITH avg_order AS (SELECT AVG(total) as avg_total FROM orders) SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id CROSS JOIN avg_order WHERE o.total > avg_order.avg_total;",
                "-- Get monthly sales report\nSELECT DATE_TRUNC('month', order_date) as month, COUNT(*) as orders, SUM(total) as revenue FROM orders WHERE order_date >= '2023-01-01' GROUP BY month ORDER BY month;"
            ]
        }
    
    def build_prompt(
        self, 
        user_request: str, 
        query_type: QueryType = QueryType.SELECT,
        schema_info: Optional[SchemaInfo] = None,
        include_examples: bool = True,
        max_examples: int = 2
    ) -> str:
        """Costruisce prompt per la generazione SQL."""
        
        # Schema info formatting
        schema_text = "No schema information provided"
        if schema_info:
            schema_parts = []
            for table, columns in schema_info.tables.items():
                schema_parts.append(f"Table: {table} (Columns: {', '.join(columns)})")
            schema_text = "\n".join(schema_parts)
        
        # Examples formatting
        examples_text = ""
        if include_examples and query_type in self.few_shot_examples:
            selected_examples = self.few_shot_examples[query_type][:max_examples]
            examples_text = "\n\n".join(selected_examples)
        
        # Get template
        template = self.base_templates.get(query_type, self.base_templates[QueryType.SELECT])
        
        # Fill template
        prompt = template.format(
            dialect=self.dialect.value.upper(),
            schema_info=schema_text,
            user_request=user_request,
            examples=examples_text
        )
        
        return prompt
    
    def detect_query_type(self, user_request: str) -> QueryType:
        """Rileva il tipo di query dalla richiesta dell'utente."""
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ["select", "find", "get", "show", "list", "count"]):
            if any(word in request_lower for word in ["join", "subquery", "cte", "window", "complex"]):
                return QueryType.COMPLEX
            return QueryType.SELECT
        elif any(word in request_lower for word in ["insert", "add", "create new"]):
            return QueryType.INSERT
        elif any(word in request_lower for word in ["update", "modify", "change"]):
            return QueryType.UPDATE
        elif any(word in request_lower for word in ["delete", "remove"]):
            return QueryType.DELETE
        elif any(word in request_lower for word in ["create table", "new table"]):
            return QueryType.CREATE_TABLE
        else:
            return QueryType.SELECT  # Default



class SQLLLMGenerator:
    """Generatore SQL usando HuggingFace Transformers."""
    
    def __init__(
        self, 
        config: SQLGenerationConfig, 
        dialect: SQLDialect = SQLDialect.POSTGRESQL,
        local_model_path: Optional[str] = None
    ):
        self.config = config
        self.dialect = dialect
        self.local_model_path = local_model_path
        
        # Inizializza prompt builder
        self.prompt_builder = SQLPromptBuilder(dialect)
        
        # Carica modello e tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self._load_model()
        
        logger.info("SQL LLM Generator inizializzato")
    
    def _load_model(self):
        """Carica il modello LLM."""
        try:
            model_path = self.local_model_path or self.config.model_name
            
            logger.info(f"Caricamento modello: {model_path}")
            
            # Carica tokenizer e modello
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Set pad token se non disponibile
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Crea pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                return_full_text=self.config.return_full_text,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Modello caricato con successo")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento modello: {str(e)}")
            raise
    
    def generate_sql(
        self,
        user_request: str,
        schema_info: Optional[SchemaInfo] = None,
        query_type: Optional[QueryType] = None
    ) -> Dict[str, Any]:
        """Genera query SQL basata sulla richiesta dell'utente."""
        
        try:
            # Auto-detect query type se non specificato
            if query_type is None:
                query_type = self.prompt_builder.detect_query_type(user_request)
            
            logger.info(f"Generazione SQL - Tipo: {query_type.value}, Request: {user_request[:100]}...")
            
            # Costruisci prompt
            prompt = self.prompt_builder.build_prompt(
                user_request=user_request,
                query_type=query_type,
                schema_info=schema_info,
                include_examples=True
            )
            
            # Genera SQL
            logger.info("Generazione in corso...")
            result = self.pipeline(prompt)
            
            # Estrai SQL dall'output
            generated_text = result[0]["generated_text"]
            sql_query = self._extract_sql_from_output(generated_text, query_type)
            
            generation_result = {
                "success": True,
                "user_request": user_request,
                "query_type": query_type.value,
                "dialect": self.dialect.value,
                "generated_sql": sql_query,
                "raw_output": generated_text,
                "prompt_used": prompt,
                "schema_info": schema_info.tables if schema_info else None
            }
            
            logger.info(f"SQL generato con successo: {sql_query[:100]}...")
            return generation_result
            
        except Exception as e:
            logger.error(f"Errore nella generazione SQL: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "user_request": user_request,
                "query_type": query_type.value if query_type else None,
                "dialect": self.dialect.value
            }
    
    def _extract_sql_from_output(self, generated_text: str, query_type: QueryType) -> str:
        """Estrae query SQL pulita dall'output generato."""
        # Rimuovi commenti e spazi extra
        lines = generated_text.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--'):
                sql_lines.append(line)
        
        sql_query = ' '.join(sql_lines).strip()
        
        # Pulizia finale
        if sql_query.endswith(';'):
            sql_query = sql_query[:-1]
        
        return sql_query
    
    
    def batch_generate(
        self,
        requests: List[str],
        schema_info: Optional[SchemaInfo] = None
    ) -> List[Dict[str, Any]]:
        """Genera SQL per multiple richieste in batch."""
        results = []
        
        for i, request in enumerate(requests):
            logger.info(f"Batch generation {i+1}/{len(requests)}: {request[:50]}...")
            
            result = self.generate_sql(
                user_request=request,
                schema_info=schema_info
            )
            
            results.append(result)
        
        return results
    
    def interactive_mode(self, schema_info: Optional[SchemaInfo] = None):
        """Modalit� interattiva per generazione SQL."""
        print("\n" + "="*60)
        print("SQL LLM GENERATOR - MODALIT� INTERATTIVA")
        print(f"Dialetto: {self.dialect.value.upper()}")
        print("Digita 'exit' per uscire, 'schema' per vedere lo schema")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n=9 SQL Request: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n=K Arrivederci!")
                    break
                
                if user_input.lower() == 'schema':
                    if schema_info:
                        print("\n=� SCHEMA INFO:")
                        for table, columns in schema_info.tables.items():
                            print(f"  Table: {table}")
                            print(f"    Columns: {', '.join(columns)}")
                    else:
                        print("\n�  Nessuna informazione di schema disponibile")
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n= Generazione SQL per: {user_input}")
                print("� Processing...")
                
                result = self.generate_sql(
                    user_request=user_input,
                    schema_info=schema_info
                )
                
                if result["success"]:
                    print(f"\n SQL GENERATO:")
                    print("="*50)
                    print(result["generated_sql"])
                    print("="*50)
                    
                    if result["validation"]:
                        if result["is_valid"]:
                            print(" Query sintatticamente valida")
                        else:
                            print("L Query non valida:")
                            print(f"   Error: {result['validation']['error']}")
                            if result['validation']['suggestions']:
                                print("   Suggestions:")
                                for suggestion in result['validation']['suggestions']:
                                    print(f"   - {suggestion}")
                else:
                    print(f"\nL Errore nella generazione: {result['error']}")
                
            except KeyboardInterrupt:
                print("\n\n=K Sessione terminata dall'utente.")
                break
            except Exception as e:
                print(f"\nL Errore: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="SQL LLM Generator con HuggingFace Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

1. Generazione SQL semplice:
   python LLMsql.py --model "microsoft/DialoGPT-medium" --query "Find all users with age > 25"

2. Con dialetto specifico:
   python LLMsql.py --model "microsoft/DialoGPT-medium" --query "Get total sales by category" --dialect postgresql

3. Con schema personalizzato:
   python LLMsql.py --model "microsoft/DialoGPT-medium" --query "Join users and orders" --schema schema.json

4. Modalit� interattiva:
   python LLMsql.py --model "microsoft/DialoGPT-medium" --interactive

5. Batch processing:
   python LLMsql.py --model "microsoft/DialoGPT-medium" --batch queries.txt

DIALETTI SUPPORTATI:
  postgresql, mysql, sqlite, bigquery, snowflake, oracle, sqlserver, redshift, clickhouse, generic

TIPI DI QUERY:
  SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, ALTER TABLE, Complex JOINs/Subqueries

FUNZIONALIT�:
  " Generazione SQL multi-dialetto
  " Validazione sintassi con SQLGlot
  " Template-based prompt engineering
  " Few-shot learning con esempi
  " Schema-aware generation
  " Interactive e batch modes
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Nome del modello HuggingFace per generazione SQL")
    parser.add_argument("--local_model_path", type=str, default=None,
                       help="Path al modello locale (sovrascrive --model)")
    parser.add_argument("--query", type=str,
                       help="Richiesta SQL in linguaggio naturale")
    parser.add_argument("--dialect", type=str, 
                       choices=[d.value for d in SQLDialect], 
                       default=SQLDialect.POSTGRESQL.value,
                       help="Dialetto SQL target")
    parser.add_argument("--schema", type=str,
                       help="Path al file JSON con schema database")
    parser.add_argument("--interactive", action="store_true",
                       help="Modalit� interattiva")
    parser.add_argument("--batch", type=str,
                       help="Path al file con richieste multiple (una per riga)")
    parser.add_argument("--validate", action="store_true", default=True,
                       help="Abilita validazione sintassi SQL")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature per generazione (default: 0.1)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Lunghezza massima generazione")
    parser.add_argument("--num_candidates", type=int, default=1,
                       help="Numero di candidati SQL da generare")
    parser.add_argument("--verbose", action="store_true",
                       help="Abilita logging verboso")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configurazione generazione
    config = SQLGenerationConfig(
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temperature,
        num_return_sequences=args.num_candidates
    )
    
    # Dialetto SQL
    dialect = SQLDialect(args.dialect)
    
    # Carica schema se fornito
    schema_info = None
    if args.schema:
        try:
            with open(args.schema, 'r') as f:
                schema_data = json.load(f)
                schema_info = SchemaInfo(
                    tables=schema_data.get("tables", {}),
                    relationships=schema_data.get("relationships"),
                    indexes=schema_data.get("indexes"),
                    constraints=schema_data.get("constraints")
                )
                logger.info(f"Schema caricato: {len(schema_info.tables)} tabelle")
        except Exception as e:
            logger.error(f"Errore nel caricamento schema: {e}")
            return
    
    # Inizializza generatore
    logger.info("=== INIZIALIZZAZIONE SQL LLM GENERATOR ===")
    logger.info(f"Modello: {args.local_model_path or args.model}")
    logger.info(f"Dialetto: {dialect.value.upper()}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max length: {args.max_length}")
    
    try:
        generator = SQLLLMGenerator(
            config=config,
            dialect=dialect,
            local_model_path=args.local_model_path
        )
        
        logger.info("=== GENERATORE SQL INIZIALIZZATO ===")
        
        if args.interactive:
            # Modalit� interattiva
            generator.interactive_mode(schema_info=schema_info)
            
        elif args.batch:
            # Modalit� batch
            try:
                with open(args.batch, 'r') as f:
                    requests = [line.strip() for line in f if line.strip()]
                
                logger.info(f"Batch processing: {len(requests)} richieste")
                
                results = generator.batch_generate(
                    requests=requests,
                    schema_info=schema_info,
                    validate=args.validate
                )
                
                print(f"\n{'='*60}")
                print("RISULTATI BATCH PROCESSING")
                print(f"{'='*60}")
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. REQUEST: {result['user_request']}")
                    if result["success"]:
                        print(f"   SQL: {result['best_sql']}")
                        if result['is_valid']:
                            print("   STATUS:  Valid")
                        else:
                            print("   STATUS: L Invalid")
                    else:
                        print(f"   ERROR: {result['error']}")
                
            except Exception as e:
                logger.error(f"Errore nel batch processing: {e}")
                
        elif args.query:
            # Singola query
            print(f"\n{'='*60}")
            print("SQL GENERATION")
            print(f"{'='*60}")
            print(f"Request: {args.query}")
            print(f"Dialect: {dialect.value.upper()}")
            
            result = generator.generate_sql(
                user_request=args.query,
                schema_info=schema_info,
                validate=args.validate,
                num_candidates=args.num_candidates
            )
            
            if result["success"]:
                print(f"\n GENERATED SQL:")
                print("="*50)
                print(result["best_sql"])
                print("="*50)
                
                if result["validation"]:
                    if result["is_valid"]:
                        print("\n Query is syntactically valid")
                    else:
                        print("\nL Query validation failed:")
                        print(f"Error: {result['validation']['error']}")
                        if result['validation']['suggestions']:
                            print("Suggestions:")
                            for suggestion in result['validation']['suggestions']:
                                print(f"  - {suggestion}")
                
                # Mostra candidati alternativi se pi� di uno
                if len(result["all_candidates"]) > 1:
                    print(f"\n= ALTERNATIVE CANDIDATES ({len(result['all_candidates'])-1}):")
                    for candidate in result["all_candidates"][1:]:
                        status = "" if candidate["is_valid"] else "L"
                        print(f"  {status} {candidate['sql_query']}")
                        
            else:
                print(f"\nL Generation failed: {result['error']}")
        
        else:
            # Nessuna query specificata
            print("\n�  Nessuna query specificata.")
            print("Usa --query 'your request' per una singola generazione")
            print("Usa --interactive per modalit� interattiva")
            print("Usa --batch file.txt per processing batch")
            print("Usa --help per vedere tutte le opzioni")
            
    except Exception as e:
        logger.error(f"Errore nell'inizializzazione: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())