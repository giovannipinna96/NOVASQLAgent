"""
Sistema agente per la consultazione di documentazione utilizzando smolagents e il server MCP Context7.
Questo agente è specializzato nella ricerca e consultazione di documentazione tecnica tramite Context7.

FUNZIONALITÀ PRINCIPALI:
1. Integrazione con Context7 MCP server per accedere alla documentazione
2. Strumenti per risolvere ID librerie e ottenere documentazione specifica
3. Agente specializzato per query di documentazione tecnica
4. Supporto per ricerca per topic, token limits e library specifiche

UTILIZZO:
python LLMdoc.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Come creare custom tools in HuggingFace Agents"
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
from smolagents import tool, CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def resolve_library_id(library_name: str) -> str:
    """
    Risolve il nome di una libreria in un Context7-compatible library ID.
    Utilizza il server MCP Context7 per trovare l'ID corretto della libreria.
    
    Args:
        library_name: Nome della libreria da cercare (es. "transformers", "pytorch", "tensorflow")
    
    Returns:
        str: L'ID compatibile Context7 della libreria trovata, o messaggio di errore
    """
    try:
        # Simulazione di integrazione con Context7 MCP
        # In un'implementazione reale, qui ci sarebbe la chiamata MCP
        
        # Mapping comuni per librerie note
        common_libraries = {
            "transformers": "/huggingface/transformers",
            "pytorch": "/pytorch/pytorch", 
            "tensorflow": "/tensorflow/tensorflow",
            "datasets": "/huggingface/datasets",
            "tokenizers": "/huggingface/tokenizers",
            "accelerate": "/huggingface/accelerate",
            "peft": "/huggingface/peft",
            "trl": "/huggingface/trl",
            "smolagents": "/huggingface/smolagents",
            "agents": "/huggingface/transformers",  # Redirect to transformers
            "numpy": "/numpy/numpy",
            "pandas": "/pandas/pandas",
            "scikit-learn": "/scikit-learn/scikit-learn",
            "matplotlib": "/matplotlib/matplotlib",
            "seaborn": "/mwaskom/seaborn",
            "flask": "/pallets/flask",
            "django": "/django/django",
            "fastapi": "/tiangolo/fastapi"
        }
        
        library_lower = library_name.lower()
        
        if library_lower in common_libraries:
            library_id = common_libraries[library_lower]
            logger.info(f"Risolto '{library_name}' -> '{library_id}'")
            return f"Libreria trovata: {library_id}"
        else:
            # Ricerca fuzzy per librerie simili
            similar = [lib for lib in common_libraries.keys() if library_lower in lib or lib in library_lower]
            if similar:
                suggestions = ", ".join([f"{lib} -> {common_libraries[lib]}" for lib in similar[:3]])
                return f"Libreria '{library_name}' non trovata direttamente. Possibili alternative: {suggestions}"
            else:
                return f"Libreria '{library_name}' non trovata. Prova con: transformers, pytorch, tensorflow, datasets, accelerate, peft, trl"
    
    except Exception as e:
        logger.error(f"Errore nella risoluzione della libreria: {str(e)}")
        return f"Errore nella risoluzione della libreria: {str(e)}"


@tool  
def get_library_documentation(library_id: str, topic: str = "", max_tokens: int = 8000) -> str:
    """
    Ottiene documentazione per una specifica libreria utilizzando Context7.
    
    Args:
        library_id: ID Context7-compatible della libreria (es. "/huggingface/transformers")
        topic: Topic specifico da cercare (opzionale, es. "agents", "custom tools", "pipelines")  
        max_tokens: Numero massimo di token di documentazione da recuperare (default: 8000)
    
    Returns:
        str: Documentazione recuperata o messaggio di errore
    """
    try:
        # Simulazione di chiamata al server MCP Context7
        # In un'implementazione reale, qui ci sarebbe la chiamata diretta MCP
        
        logger.info(f"Recuperando documentazione per {library_id}, topic: '{topic}', max_tokens: {max_tokens}")
        
        # Documentazione di esempio basata sui risultati precedenti di Context7
        if "/huggingface/transformers" in library_id:
            if "agents" in topic.lower() or "smolagents" in topic.lower():
                return """
DOCUMENTAZIONE HUGGINGFACE AGENTS/SMOLAGENTS:

La funzionalità Agents è stata spostata nella libreria standalone 'smolagents' a partire dalla versione v4.52 di transformers.

MIGRAZIONE A SMOLAGENTS:
- Installazione: pip install smolagents
- Documentazione: https://huggingface.co/docs/smolagents/index

CREAZIONE CUSTOM TOOLS:
1. Usa il decoratore @tool:
   ```python
   from smolagents import tool
   
   @tool
   def my_custom_tool(parameter: str) -> str:
       '''Descrizione del tool'''
       return result
   ```

2. Oppure eredita dalla classe Tool:
   ```python  
   from smolagents import Tool
   
   class MyCustomTool(Tool):
       name = "my_tool"
       description = "Descrizione del tool"
       
       def forward(self, parameter: str) -> str:
           return result
   ```

INTEGRAZIONE CON CODEAGENT:
```python
from smolagents import CodeAgent, TransformersModel

model = TransformersModel(model_id="your-model")
agent = CodeAgent(tools=[my_custom_tool], model=model)
result = agent.run("your query")
```

MCP SERVER INTEGRATION:
- Configurazione tiny-agents per MCP: {"servers": [{"type": "sse", "url": "your-mcp-endpoint"}]}
- Supporto per server SSE e WebSocket
"""
            
            elif "custom" in topic.lower() and "pipeline" in topic.lower():
                return """
DOCUMENTAZIONE CUSTOM PIPELINES TRANSFORMERS:

CREAZIONE PIPELINE PERSONALIZZATE:
1. Eredita dalla classe Pipeline base:
   ```python
   from transformers import Pipeline
   
   class MyCustomPipeline(Pipeline):
       def _sanitize_parameters(self, **kwargs):
           # Gestisce parametri input
           return preprocess_kwargs, forward_kwargs, postprocess_kwargs
           
       def preprocess(self, inputs, **preprocess_kwargs):
           # Preprocessa input
           return processed_inputs
           
       def _forward(self, model_inputs, **forward_kwargs):
           # Forward pass attraverso il modello  
           return model_outputs
           
       def postprocess(self, model_outputs, **postprocess_kwargs):
           # Post-processa output
           return final_outputs
   ```

2. Registrazione pipeline:
   ```python
   from transformers.pipelines import PIPELINE_REGISTRY
   
   PIPELINE_REGISTRY.register_pipeline(
       "my-task",
       pipeline_class=MyCustomPipeline,
       pt_model=AutoModel,
   )
   ```

3. Utilizzo:
   ```python
   from transformers import pipeline
   my_pipeline = pipeline("my-task", model="model-name")
   ```
"""
            
            else:
                return """
DOCUMENTAZIONE GENERICA HUGGINGFACE TRANSFORMERS:

CARATTERISTICHE PRINCIPALI:
- Modelli pre-addestrati per NLP, Vision, Audio
- Supporto PyTorch, TensorFlow, JAX  
- Pipeline per task comuni
- Fine-tuning e training personalizzato
- Quantizzazione e ottimizzazioni

COMPONENTI PRINCIPALI:
- AutoModel/AutoTokenizer per caricamento automatico modelli
- Trainer per training e fine-tuning
- Pipeline per inferenza rapida
- Configurazioni personalizzabili
- Integrazione con Hub

UTILIZZO BASE:
```python
from transformers import AutoModel, AutoTokenizer, pipeline

# Pipeline rapida
classifier = pipeline("text-classification", model="model-name")

# Controllo manuale  
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")
```

Per topic specifici, specifica nell'argomento 'topic': agents, pipelines, custom models, training, etc.
"""
        
        elif "/pytorch" in library_id:
            return """
DOCUMENTAZIONE PYTORCH:

PyTorch è una libreria di deep learning con supporto per:
- Tensor computing con GPU acceleration
- Automatic differentiation
- Neural network building blocks
- Distributed training
- Mobile deployment

COMPONENTI PRINCIPALI:
- torch: tensor operations
- torch.nn: neural network layers
- torch.optim: optimization algorithms  
- torch.utils.data: dataset utilities
- torchvision: computer vision utilities
"""
        
        else:
            return f"""
Documentazione richiesta per: {library_id}
Topic: {topic}
Max tokens: {max_tokens}

NOTA: Questa è una simulazione. In un'implementazione reale, qui verrebbe effettuata
una chiamata al server MCP Context7 per recuperare la documentazione aggiornata
dalla fonte specificata.

Per librerie supportate, prova: /huggingface/transformers, /pytorch/pytorch, ecc.
"""
    
    except Exception as e:
        logger.error(f"Errore nel recupero documentazione: {str(e)}")
        return f"Errore nel recupero documentazione: {str(e)}"


@tool
def search_documentation_topic(query: str, libraries: str = "transformers") -> str:
    """
    Cerca informazioni specifiche nella documentazione per un topic o query.
    
    Args:
        query: Query di ricerca (es. "how to create custom tools", "pipeline registration")
        libraries: Librerie in cui cercare, separate da virgola (default: "transformers")
    
    Returns:
        str: Risultati di ricerca pertinenti o messaggio di errore
    """
    try:
        logger.info(f"Cercando '{query}' nelle librerie: {libraries}")
        
        # Logica di ricerca semplificata
        query_lower = query.lower()
        lib_list = [lib.strip().lower() for lib in libraries.split(",")]
        
        results = []
        
        for lib in lib_list:
            # Risolvi libreria
            lib_id_result = resolve_library_id(lib)
            if "Libreria trovata:" in lib_id_result:
                lib_id = lib_id_result.split(": ")[1]
                
                # Determina topic basato sulla query
                if any(word in query_lower for word in ["agent", "tool", "smolagent"]):
                    topic = "agents smolagents custom tools"
                elif any(word in query_lower for word in ["pipeline", "custom pipeline"]):
                    topic = "pipelines custom"
                elif any(word in query_lower for word in ["training", "trainer", "fine-tune"]):
                    topic = "training fine-tuning"
                elif any(word in query_lower for word in ["model", "custom model"]):
                    topic = "custom models"
                else:
                    topic = query
                
                # Ottieni documentazione
                doc_result = get_library_documentation(lib_id, topic, 5000)
                results.append(f"=== RISULTATI PER {lib.upper()} ===\n{doc_result}\n")
        
        return "\n".join(results) if results else "Nessun risultato trovato per la query specificata."
    
    except Exception as e:
        logger.error(f"Errore nella ricerca documentazione: {str(e)}")
        return f"Errore nella ricerca: {str(e)}"


@tool
def list_supported_libraries() -> str:
    """
    Elenca le librerie supportate dal sistema di documentazione Context7.
    
    Returns:
        str: Lista delle librerie supportate con i loro ID Context7
    """
    supported = {
        "HuggingFace Ecosystem": [
            ("transformers", "/huggingface/transformers"),
            ("datasets", "/huggingface/datasets"), 
            ("tokenizers", "/huggingface/tokenizers"),
            ("accelerate", "/huggingface/accelerate"),
            ("peft", "/huggingface/peft"),
            ("trl", "/huggingface/trl"),
            ("smolagents", "/huggingface/smolagents")
        ],
        "Deep Learning Frameworks": [
            ("pytorch", "/pytorch/pytorch"),
            ("tensorflow", "/tensorflow/tensorflow")
        ],
        "Data Science": [
            ("numpy", "/numpy/numpy"),
            ("pandas", "/pandas/pandas"),
            ("scikit-learn", "/scikit-learn/scikit-learn"),
            ("matplotlib", "/matplotlib/matplotlib"),
            ("seaborn", "/mwaskom/seaborn")
        ],
        "Web Frameworks": [
            ("flask", "/pallets/flask"),
            ("django", "/django/django"),
            ("fastapi", "/tiangolo/fastapi")
        ]
    }
    
    result = "=== LIBRERIE SUPPORTATE ===\n\n"
    for category, libs in supported.items():
        result += f"{category}:\n"
        for name, lib_id in libs:
            result += f"  • {name} -> {lib_id}\n"
        result += "\n"
    
    result += "UTILIZZO:\n"
    result += "1. resolve_library_id('nome_libreria') - per ottenere l'ID Context7\n"
    result += "2. get_library_documentation('id_libreria', 'topic') - per ottenere documentazione\n"
    result += "3. search_documentation_topic('query', 'librerie') - per ricerca avanzata\n"
    
    return result


def build_llm(model_id: str, local_model_path: str = None):
    """Costruisce il modello LLM per l'agente di documentazione"""
    if local_model_path:
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="LLMdoc: Agente specializzato per consultazione documentazione tecnica via Context7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

1. Ricerca generale:
   python LLMdoc.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Come creare custom tools in HuggingFace Agents"

2. Documentazione specifica:
   python LLMdoc.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Mostrami la documentazione di transformers sui custom pipelines"

3. Ricerca multi-libreria:  
   python LLMdoc.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Confronta PyTorch e TensorFlow per training distribuito"

4. Lista librerie supportate:
   python LLMdoc.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Quali librerie sono supportate dal sistema di documentazione?"

STRUMENTI DISPONIBILI:
  • resolve_library_id: Risolve nomi librerie in ID Context7
  • get_library_documentation: Recupera documentazione specifica  
  • search_documentation_topic: Ricerca avanzata per topic
  • list_supported_libraries: Lista librerie supportate

INTEGRAZIONE MCP CONTEXT7:
  Questo agente è progettato per integrarsi con il server MCP Context7, fornendo
  accesso aggiornato alla documentazione tecnica di librerie e framework.
        """
    )
    
    parser.add_argument("--model", type=str, required=True, 
                       help="Nome del modello HuggingFace per l'agente")
    parser.add_argument("--query", type=str, required=True,
                       help="Query o domanda sulla documentazione") 
    parser.add_argument("--local_model_path", type=str, default=None,
                       help="Path al modello locale (sovrascrive --model se fornito)")
    parser.add_argument("--max_tokens", type=int, default=8000,
                       help="Numero massimo di token per la documentazione")
    parser.add_argument("--verbose", action="store_true",
                       help="Abilita logging verboso")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== INIZIALIZZAZIONE LLMDOC AGENT ===")
    logger.info(f"Modello: {args.local_model_path or args.model}")
    logger.info(f"Query: {args.query}")
    
    # Carica modello LLM  
    model = TransformersModel(model_id=args.local_model_path or args.model)
    
    # Crea agente con strumenti di documentazione
    documentation_tools = [
        resolve_library_id,
        get_library_documentation, 
        search_documentation_topic,
        list_supported_libraries
    ]
    
    agent = CodeAgent(tools=documentation_tools, model=model)
    
    logger.info("=== PROCESSING QUERY ===")
    
    # Esegui query
    result = agent.run(args.query)
    
    print(f"\n" + "="*60)
    print("LLMDOC AGENT - RISULTATO CONSULTAZIONE DOCUMENTAZIONE")
    print("="*60)
    print(f"QUERY: {args.query}")
    print("-"*60) 
    print(f"RISPOSTA:\n{result}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()