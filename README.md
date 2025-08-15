# NOVASQLAgent

## 🚀 Introduzione

NOVASQLAgent è un sistema avanzato di intelligenza artificiale multi-agente progettato per la comprensione, traduzione e ottimizzazione di query SQL. Il sistema combina architetture LLM multiple, pipeline di training e un sistema completo di generazione SQL che elabora prompt esterni attraverso diverse fasi per generare query SQL ottimizzate.

Il progetto nasce per affrontare i benchmark più avanzati nel campo del Text-to-SQL e dell'automazione di pipeline ELT, tra cui **Spider 2.0** e **ELT-Bench**, fornendo un'architettura scalabile e modulare per la generazione automatica di codice SQL enterprise-grade.

## 📋 Descrizione del Progetto

### Caratteristiche Principali

- **🤖 Sistema Multi-Agente**: Orchestrazione di agenti specializzati per diverse funzioni (SQL generation, RAG, planning, optimization)
- **🕷️ Compatibilità Spider 2.0**: Supporto completo per tutti i variant del benchmark Spider 2.0 (Lite, Snow, DBT)
- **🔄 Integrazione ELT-Bench**: Elaborazione nativa di task ELT-Bench per automazione pipeline
- **🔧 Pipeline a 7 Fasi**: Workflow strutturato da configurazione a output finale
- **🌍 Multi-Dialect SQL**: Supporto per 24+ dialetti SQL (PostgreSQL, MySQL, BigQuery, Snowflake, etc.)
- **⚡ Processing Parallelo**: Elaborazione concorrente per performance ottimali
- **📊 Benchmark Evaluation**: Sistema completo per valutazione su dataset reali

### Architettura del Sistema

```
NOVASQLAgent/
├── 🎯 Core Pipeline (main.py)           # Interfaccia CLI semplice
├── 🕷️ Spider Agent (spider_agent/)      # Sistema compatibile ELT-Bench
├── 🤖 Multi-Agent System (final.py)     # Orchestratore completo
├── 📊 Benchmark Processor (final2.py)   # Elaborazione dataset reali
├── 🧠 Model Components (src/model/)     # Agenti LLM specializzati
├── 🔄 SQL Translation (src/SQLTranlator/) # Traduzione multi-dialect
├── 🗄️ Vector Database (src/vectorDB/)   # Sistema RAG con FAISS
└── 📈 Training Pipeline (src/train/)    # Fine-tuning SFT e GRPO
```

## 🗂️ Descrizione dei File e Componenti

### 📁 File Principali di Esecuzione

#### 1. `main.py` - Pipeline CLI Diretta
**Scopo**: Interfaccia a riga di comando semplice per generazione SQL rapida.

**Funzionalità**:
- Generazione SQL da prompt e descrizioni
- Traduzione multi-dialect
- Validazione sintassi SQL
- Configurazione via CLI

**Come eseguire**:
```bash
# Esempio base
python main.py --prompt "Find all users with age > 25" --descriptions "users table contains id, name, age columns" --output query.sql

# Con dialect specifico
python main.py --prompt "Get monthly sales" --descriptions "sales table, products table" --dialect postgresql

# Configurazione avanzata
python main.py --prompt "Complex analytics query" --descriptions "desc1" "desc2" --variants 5 --confidence 0.8 --dialect bigquery

# Modalità dry-run per testing
python main.py --prompt "Test query" --descriptions "test table" --dry-run --verbose
```

#### 2. `final.py` - Sistema Multi-Agente Coordinatore
**Scopo**: Orchestratore completo che coordina tutti i componenti per task complessi.

**Funzionalità**:
- Auto-detection tipo benchmark (Spider 2.0 vs ELT-Bench)
- Orchestrazione multi-agente (RAG, Planning, Optimization)
- Processing parallelo di batch task
- Sistema di reporting completo

**Come eseguire**:
```bash
# Esecuzione con task di esempio
python final.py --example-tasks --verbose

# Processing parallelo con agenti multipli
python final.py --input-file tasks.json --parallel --max-agents 5

# Benchmark specifico con modello personalizzato
python final.py --benchmark spider2-lite --model gpt-4o --output-dir ./results

# Configurazione completa
python final.py --benchmark auto --parallel --max-agents 3 --enable-rag --enable-planning --verbose
```

#### 3. `final2.py` - Processore Dataset Reali
**Scopo**: Elabora dataset reali di Spider 2.0 e ELT-Bench per valutazione benchmark.

**Funzionalità**:
- Caricamento automatico dataset dalla cartella `data/`
- Generazione file di submission per benchmark
- Scoring e valutazione automatica
- Report di performance dettagliati

**Come eseguire**:
```bash
# Processing di tutti i benchmark disponibili
python final2.py --data-dir ./data --output-dir ./final2_results --verbose

# Solo Spider 2.0 Lite con concorrenza limitata
python final2.py --benchmark spider2-lite --max-concurrent 3 --limit 50

# ELT-Bench con debug completo
python final2.py --benchmark elt-bench --debug --max-concurrent 2

# Processing rapido per testing
python final2.py --benchmark spider2-lite --limit 10 --verbose
```

#### 4. `example_usage.py` - Esempi Dimostrativi
**Scopo**: Esempi progressivi per apprendimento e testing del sistema.

**Come eseguire**:
```bash
# Tutti gli esempi progressivi
python example_usage.py

# Solo esempio base
python -c "from example_usage import basic_example; basic_example()"

# Solo esempio avanzato
python -c "from example_usage import advanced_example; advanced_example()"
```

### 🧠 Componenti Modello (src/model/)

#### `LLMasJudge.py` - Sistema di Giudizio e Riscrittura
**Funzionalità**: Giudizio di rilevanza e riscrittura prompt con token probability analysis.

**Esecuzione standalone**:
```bash
# Test modalità giudizio
python src/model/LLMasJudge.py --mode judge --text1 "SQL query text" --text2 "Table description"

# Test modalità riscrittura
python src/model/LLMasJudge.py --mode rewrite --text1 "Original prompt"

# Test entrambe le modalità
python src/model/LLMasJudge.py --mode both --text1 "Sample text"
```

#### `LLMsql.py` - Generatore SQL Avanzato
**Funzionalità**: Generazione SQL multi-dialect con template e few-shot learning.

**Esecuzione standalone**:
```bash
# Generazione base
python src/model/LLMsql.py --model "microsoft/DialoGPT-medium" --query "Find all users with age > 25"

# Con dialect specifico
python src/model/LLMsql.py --query "Monthly sales report" --dialect postgresql

# Modalità interattiva
python src/model/LLMsql.py --interactive
```

#### `LLMmerge.py` - Sistema di Merge Query
**Funzionalità**: Combinazione intelligente di query multiple usando strategie diverse.

**Esecuzione standalone**:
```bash
# Test merge con query multiple
python src/model/LLMmerge.py --queries "SELECT * FROM users" "SELECT * FROM orders"

# Modalità dettagliata
python src/model/LLMmerge.py --detailed --input "query1.sql" "query2.sql"
```

#### `LLMrag.py` - Agente RAG con Vector Database
**Funzionalità**: Retrieval-Augmented Generation con FAISS e gestione conversazioni.

**Esecuzione standalone**:
```bash
# Modalità interattiva RAG
python src/model/LLMrag.py --model "microsoft/DialoGPT-medium" --query "Explain machine learning" --interactive

# Ricerca con contesto
python src/model/LLMrag.py --query "SQL best practices" --context "database optimization"
```

#### `LLMpromptopt.py` - Ottimizzazione Prompt
**Funzionalità**: Miglioramento automatico prompt con 6 strategie diverse.

**Esecuzione standalone**:
```bash
# Ottimizzazione singolo prompt
python src/model/LLMpromptopt.py --prompt "Find users" --strategy detailed_expansion

# Batch optimization
python src/model/LLMpromptopt.py --batch --input-file prompts.txt

# Tutte le strategie
python src/model/LLMpromptopt.py --prompt "Complex query" --all-strategies
```

#### `AgentHF.py` - Agente Bash e Code Execution
**Funzionalità**: Esecuzione comandi bash e operazioni filesystem.

**Esecuzione standalone**:
```bash
# Esecuzione query matematica
python src/model/AgentHF.py --query "Calculate 5 + 3 * 2"

# Comando bash
python src/model/AgentHF.py --command "ls -la"
```

### 🔄 Componenti SQL e Database

#### `SQLTranslator` - Traduzione Multi-Dialect
**Esecuzione standalone**:
```bash
# Test traduzione
python src/SQLTranlator/sql_translator.py

# Traduzione specifica
python -c "from src.SQLTranlator.sql_translator import SQLTranslator; t=SQLTranslator(); print(t.translate('SELECT * FROM users', 'postgresql', 'mysql'))"
```

#### `VectorDB Manager` - Gestione Database Vettoriale
**Esecuzione standalone**:
```bash
# Demo completa
python src/vectorDB/vectorDB_manager.py --demo

# Test ricerca semantica
python -c "from src.vectorDB.vectorDB_manager import VectorDBManager; vdb=VectorDBManager(); vdb.test_basic_operations()"
```

## 📊 Sistema di Gestione Dati

### Struttura Directory Dati

```
data/
├── spider2-lite/
│   ├── questions.jsonl          # Task Spider 2.0 Lite
│   └── databases/               # Schema database
├── spider2-snow/
│   ├── snowflake_queries.json   # Query Snowflake-specific
│   └── schemas/                 # Schema Snowflake
├── spider2-dbt/
│   ├── dbt_tasks.json          # Task DBT modeling
│   └── models/                 # Modelli DBT
└── elt-bench/
    ├── tasks.json              # Task ELT pipeline
    └── data_streams/           # Stream di dati esempio
```

### Download e Setup Dataset

#### 1. Spider 2.0 Dataset
```bash
# Download automatico Spider 2.0
python src/data/raw_data_manager.py --dataset spider2 --output-dir ./data

# Oppure download manuale da GitHub
wget https://github.com/spider-benchmark/spider2.0/archive/main.zip
unzip -d data/spider2-lite/
```

#### 2. ELT-Bench Dataset
```bash
# Download ELT-Bench
python src/data/raw_data_manager.py --dataset elt-bench --output-dir ./data

# Setup manuale
git clone https://github.com/elt-bench/elt-bench.git data/elt-bench
```

#### 3. Training Dataset
```bash
# Download dataset per training
python src/data/raw_data_manager.py

# Dataset disponibili:
# - ARCHER-BENCH
# - BIRD
# - SynSQL-2.5M
```

### Uso del Sistema Dati

#### Verifica Dataset
```bash
# Controlla dataset disponibili
python final2.py --data-dir ./data --verbose --limit 0

# Statistiche dataset
python -c "from final2 import RealBenchmarkProcessor; p=RealBenchmarkProcessor('./data'); tasks=p.discover_benchmark_data(); print(f'Found {len(tasks)} tasks')"
```

#### Processing Dataset Specifici
```bash
# Solo Spider 2.0 Lite (primi 10 task)
python final2.py --benchmark spider2-lite --limit 10

# Solo ELT-Bench con debug
python final2.py --benchmark elt-bench --debug --max-concurrent 1

# Tutti i dataset con processing parallelo
python final2.py --max-concurrent 5 --verbose
```

## 🏃‍♂️ Esecuzione Pipeline Completa su Benchmark

### Setup Iniziale

1. **Installazione dipendenze**:
```bash
pip install -r requirements.txt
```

2. **Download dataset benchmark**:
```bash
# Spider 2.0
python src/data/raw_data_manager.py --dataset spider2 --output-dir ./data

# ELT-Bench  
python src/data/raw_data_manager.py --dataset elt-bench --output-dir ./data
```

3. **Verifica installazione**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Esecuzione Pipeline Completa

#### Scenario 1: Test Rapido con Esempi
```bash
# 1. Test componenti base
python example_usage.py

# 2. Test sistema multi-agente
python final.py --example-tasks --verbose

# 3. Verifica output
ls -la final_results/
```

#### Scenario 2: Valutazione Spider 2.0 Completa
```bash
# 1. Processing completo Spider 2.0 Lite
python final2.py --benchmark spider2-lite --verbose --max-concurrent 3

# 2. Verifica risultati
ls -la final2_results/spider2-lite/

# 3. Analisi report
python -c "import json; report=json.load(open('final2_results/benchmark_report.json')); print(f'Success Rate: {report[\"summary\"][\"overall_success_rate\"]:.1f}%')"

# 4. Controllo file submission
head -20 final2_results/spider2-lite/spider2_lite_001.sql
```

#### Scenario 3: Pipeline ELT-Bench Completa
```bash
# 1. Processing ELT-Bench
python final2.py --benchmark elt-bench --debug --max-concurrent 2

# 2. Verifica submission files
ls -la final2_results/elt-bench/*_submission.json

# 3. Analisi performance
python -c "import json; import glob; files=glob.glob('final2_results/elt-bench/*_submission.json'); scores=[json.load(open(f))['score'] for f in files]; print(f'Average Score: {sum(scores)/len(scores):.3f}')"
```

#### Scenario 4: Benchmark Comparison
```bash
# 1. Processing tutti i benchmark
python final2.py --verbose --max-concurrent 3

# 2. Generazione report comparativo
python -c "
import json
report = json.load(open('final2_results/benchmark_report.json'))
for benchmark, stats in report['benchmark_statistics'].items():
    print(f'{benchmark}: {stats[\"success_rate\"]:.1f}% success, {stats[\"avg_score\"]:.3f} avg score')
"

# 3. Analisi errori comuni
python -c "
import json
report = json.load(open('final2_results/benchmark_report.json'))
for error, count in report['error_analysis']['common_errors'][:5]:
    print(f'{count}x: {error}')
"
```

### Output Attesi

#### Struttura Output
```
final2_results/
├── spider2-lite/
│   ├── spider2_lite_001.sql              # ✅ File submission
│   ├── spider2_lite_001_result.json      # 📊 Risultato dettagliato
│   └── ...
├── elt-bench/
│   ├── elt_bench_001_submission.json     # ✅ Submission ELT
│   ├── elt_bench_001_result.json         # 📊 Risultato dettagliato
│   └── ...
└── benchmark_report.json                 # 📈 Report finale
```

#### Metriche di Successo

- **Success Rate**: > 80% per Spider 2.0 Lite
- **Avg Execution Time**: < 30s per task
- **Benchmark Score**: > 0.7 per ELT-Bench
- **Error Rate**: < 10% failure rate

### Troubleshooting Comune

#### Problemi Dataset
```bash
# Dataset non trovato
python final2.py --data-dir ./data --verbose --limit 0  # Verifica discovery

# Formato dataset incorreto  
python -c "import json; print(json.load(open('data/spider2-lite/questions.jsonl')))"  # Test caricamento
```

#### Problemi Performance
```bash
# Troppa concorrenza
python final2.py --max-concurrent 1  # Ridurre concorrenza

# Memory issues
python final2.py --limit 10  # Limitare numero task
```

#### Problemi Modelli
```bash
# Model loading issues
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/phi-4-mini-instruct')"

# GPU/CPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Quick Start per Principianti

1. **Clone e setup**:
```bash
git clone <repo>
cd NOVASQLAgent
pip install -r requirements.txt
```

2. **Test rapido**:
```bash
python example_usage.py
```

3. **Primo benchmark**:
```bash
python final.py --example-tasks
```

4. **Dataset reale**:
```bash
python src/data/raw_data_manager.py
python final2.py --limit 5
```

## 📚 Documentazione Completa

- **`CLAUDE.md`** - Guida per sviluppatori e istruzioni Claude Code
- **`DETAILED_DOCUMENTATION.md`** - 📖 **Documentazione tecnica dettagliata** con analisi completa di:
  - Architettura interna di ogni componente
  - Esempi d'uso avanzati e configurazioni
  - Workflow completi per ogni file principale
  - Troubleshooting e best practices
  - Integrazione tra componenti
- **`README.md`** - Panoramica generale del progetto

Per una comprensione approfondita del sistema, si raccomanda di consultare `DETAILED_DOCUMENTATION.md`.