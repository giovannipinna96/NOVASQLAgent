"""
Sistema RAG (Retrieval-Augmented Generation) utilizzando HuggingFace Agents con smolagents.

FUNZIONALIT� RAG IMPLEMENTATE:
1. Agente CodeAgent con tool per database vettoriale FAISS
2. Ricerca semantica su knowledge base esistente
3. Generazione aumentata da retrieval con contesto
4. Supporto per query multi-step e reasoning
5. Tools specializzati per retrieval e context management

ARCHITETTURA:
- VectorDBManager: Database vettoriale FAISS (gi� esistente)
- CodeAgent: Agente smolagents per orchestrazione RAG
- RAG Tools: Strumenti specializzati per retrieval e generazione
- Context Management: Gestione contesto e memoria conversazionale

UTILIZZO:
python LLMrag.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Spiega il machine learning"
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

# Importa smolagents per CodeAgent
from smolagents import tool, CodeAgent, TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Aggiungi path per il database vettoriale
sys.path.append(str(Path(__file__).parent.parent))
from vectorDB.vectorDB_manager import VectorDBManager

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG system with optimized defaults."""
    vector_db_path: str = "./vector_db"
    max_retrieved_docs: int = 5
    context_window_size: int = 2048
    similarity_threshold: float = 0.3
    use_reranking: bool = True
    combine_strategy: str = "concatenate"  # "concatenate", "summarize", "weighted"


class RAGContextManager:
    """Manages conversational context for the RAG system with history tracking.
    
    Implements Observer Pattern to track conversation state and context evolution.
    Provides efficient memory management with configurable history limits.
    """
    
    def __init__(self, max_history: int = 10):
        self.conversation_history: List[Dict[str, str]] = []
        self.retrieved_contexts: List[Dict[str, Any]] = []
        self.max_history = max_history
        
    def add_turn(self, user_query: str, retrieved_docs: List[Dict], response: str):
        """Aggiunge un turno conversazionale."""
        turn = {
            "user_query": user_query,
            "retrieved_docs": retrieved_docs,
            "response": response,
            "timestamp": str(np.datetime64('now'))
        }
        
        self.conversation_history.append(turn)
        self.retrieved_contexts.extend(retrieved_docs)
        
        # Mantieni solo gli ultimi N turni
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            
    def get_context_summary(self) -> str:
        """Restituisce un riassunto del contesto conversazionale."""
        if not self.conversation_history:
            return "Nessun contesto conversazionale precedente."
            
        summary_parts = []
        for turn in self.conversation_history[-3:]:  # Ultimi 3 turni
            summary_parts.append(f"Q: {turn['user_query'][:100]}...")
            summary_parts.append(f"R: {turn['response'][:100]}...")
            
        return "\\n".join(summary_parts)


# Inizializza manager globali
rag_config = RAGConfig()
context_manager = RAGContextManager()
vector_db: Optional[VectorDBManager] = None


@tool
def initialize_vector_database(db_path: str = "./vector_db") -> str:
    """
    Inizializza e carica il database vettoriale FAISS.
    
    Args:
        db_path: Percorso al database vettoriale
        
    Returns:
        str: Stato dell'inizializzazione
    """
    global vector_db, rag_config
    
    try:
        rag_config.vector_db_path = db_path
        
        # Carica database esistente
        vector_db = VectorDBManager(
            fields=["title", "content", "category", "author"],
            db_path=db_path,
            metric="cosine"
        )
        
        # Verifica che il database contenga dati
        stats = vector_db.get_stats()
        
        if stats["total_records"] == 0:
            return f"ATTENZIONE: Database vettoriale vuoto in {db_path}. Verificare che esistano dati salvati."
            
        logger.info(f"Database vettoriale caricato: {stats}")
        
        return f"Database vettoriale inizializzato con successo. Record: {stats['total_records']}, Dimensioni: {stats['dimension']}, Metrica: {stats['metric']}"
        
    except Exception as e:
        error_msg = f"Errore nell'inizializzazione del database vettoriale: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def semantic_search(query: str, max_results: int = 5, fields: Optional[str] = None) -> str:
    """
    Esegue ricerca semantica nel database vettoriale.
    
    Args:
        query: Query di ricerca semantica
        max_results: Numero massimo di risultati (default: 5)
        fields: Campi specifici su cui cercare (separati da virgola, es: "title,content")
        
    Returns:
        str: Risultati della ricerca formattati
    """
    global vector_db
    
    if not vector_db:
        return "Errore: Database vettoriale non inizializzato. Usa 'initialize_vector_database' prima."
        
    try:
        # Parsing campi opzionali
        search_fields = None
        if fields:
            search_fields = [f.strip() for f in fields.split(",")]
            
        # Esegui ricerca semantica
        results = vector_db.search(
            query=query,
            top_k=max_results,
            fields=search_fields
        )
        
        if not results:
            return f"Nessun risultato trovato per la query: '{query}'"
            
        # Formatta risultati
        formatted_results = []
        formatted_results.append(f"=== RISULTATI RICERCA SEMANTICA ===")
        formatted_results.append(f"Query: {query}")
        formatted_results.append(f"Risultati trovati: {len(results)}\\n")
        
        for i, result in enumerate(results, 1):
            meta = result.get("meta", {})
            score = result.get("score", 0)
            
            formatted_results.append(f"{i}. DOCUMENTO {result.get('id', 'N/A')}")
            formatted_results.append(f"   Score: {score:.4f}")
            
            if "title" in meta:
                formatted_results.append(f"   Titolo: {meta['title']}")
            if "content" in meta:
                content_preview = meta['content'][:200] + "..." if len(meta['content']) > 200 else meta['content']
                formatted_results.append(f"   Contenuto: {content_preview}")
            if "category" in meta:
                formatted_results.append(f"   Categoria: {meta['category']}")
            if "author" in meta:
                formatted_results.append(f"   Autore: {meta['author']}")
                
            # Se ricerca per campo specifico
            if "field" in result:
                formatted_results.append(f"   Campo: {result['field']}")
                
            formatted_results.append("")
            
        return "\\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Errore nella ricerca semantica: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def keyword_search(query: str, fields: Optional[str] = None, exact: bool = False, logic: str = "or") -> str:
    """
    Esegue ricerca per keyword nel database.
    
    Args:
        query: Termine di ricerca
        fields: Campi in cui cercare (separati da virgola)
        exact: Se True, ricerca match esatto
        logic: Logica per campi multipli ("or" o "and")
        
    Returns:
        str: Risultati della ricerca keyword
    """
    global vector_db
    
    if not vector_db:
        return "Errore: Database vettoriale non inizializzato. Usa 'initialize_vector_database' prima."
        
    try:
        # Parsing campi
        search_fields = None
        if fields:
            search_fields = [f.strip() for f in fields.split(",")]
            
        # Esegui ricerca keyword
        results = vector_db.keyword_search(
            query=query,
            fields=search_fields,
            exact=exact,
            logic=logic
        )
        
        if not results:
            return f"Nessun risultato trovato per la keyword: '{query}'"
            
        # Formatta risultati
        formatted_results = []
        formatted_results.append(f"=== RISULTATI RICERCA KEYWORD ===")
        formatted_results.append(f"Query: {query}")
        formatted_results.append(f"Tipo: {'Esatta' if exact else 'Parziale'}")
        formatted_results.append(f"Logica: {logic.upper()}")
        formatted_results.append(f"Risultati: {len(results)}\\n")
        
        for i, result in enumerate(results, 1):
            meta = result.get("meta", {})
            matched_fields = result.get("matched_fields", [])
            
            formatted_results.append(f"{i}. DOCUMENTO {result.get('id', 'N/A')}")
            formatted_results.append(f"   Campi corrispondenti: {', '.join(matched_fields)}")
            
            if "title" in meta:
                formatted_results.append(f"   Titolo: {meta['title']}")
            if "category" in meta:
                formatted_results.append(f"   Categoria: {meta['category']}")
            if "author" in meta:
                formatted_results.append(f"   Autore: {meta['author']}")
                
            formatted_results.append("")
            
        return "\\n".join(formatted_results)
        
    except Exception as e:
        error_msg = f"Errore nella ricerca keyword: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_document_by_id(doc_id: str) -> str:
    """
    Recupera un documento specifico tramite ID.
    
    Args:
        doc_id: ID del documento da recuperare
        
    Returns:
        str: Contenuto completo del documento
    """
    global vector_db
    
    if not vector_db:
        return "Errore: Database vettoriale non inizializzato. Usa 'initialize_vector_database' prima."
        
    try:
        record = vector_db.get_record(doc_id)
        
        if not record:
            return f"Documento con ID '{doc_id}' non trovato."
            
        # Formatta documento completo
        formatted_doc = []
        formatted_doc.append(f"=== DOCUMENTO {doc_id} ===")
        
        fields = record.fields
        if "title" in fields:
            formatted_doc.append(f"Titolo: {fields['title']}")
        if "author" in fields:
            formatted_doc.append(f"Autore: {fields['author']}")
        if "category" in fields:
            formatted_doc.append(f"Categoria: {fields['category']}")
        if "year" in fields:
            formatted_doc.append(f"Anno: {fields['year']}")
            
        formatted_doc.append("")
        
        if "content" in fields:
            formatted_doc.append("CONTENUTO:")
            formatted_doc.append(fields['content'])
        
        return "\\n".join(formatted_doc)
        
    except Exception as e:
        error_msg = f"Errore nel recupero documento: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def create_rag_context(query: str, max_docs: int = 3) -> str:
    """
    Crea un contesto RAG recuperando documenti rilevanti per una query.
    
    Args:
        query: Query dell'utente
        max_docs: Numero massimo di documenti da recuperare
        
    Returns:
        str: Contesto formattato per generazione RAG
    """
    global vector_db, context_manager
    
    if not vector_db:
        return "Errore: Database vettoriale non inizializzato. Usa 'initialize_vector_database' prima."
        
    try:
        # Ricerca semantica per documenti rilevanti
        search_results = vector_db.search(query=query, top_k=max_docs)
        
        if not search_results:
            return f"Nessun documento rilevante trovato per: '{query}'"
            
        # Filtro per soglia di similarit�
        threshold = rag_config.similarity_threshold
        filtered_results = [r for r in search_results if r.get("score", 0) >= threshold]
        
        if not filtered_results:
            return f"Nessun documento supera la soglia di similarit� ({threshold}) per: '{query}'"
            
        # Costruisci contesto RAG
        context_parts = []
        context_parts.append("=== CONTESTO RECUPERATO PER RAG ===")
        context_parts.append(f"Query: {query}")
        context_parts.append(f"Documenti recuperati: {len(filtered_results)}\\n")
        
        retrieved_docs = []
        
        for i, result in enumerate(filtered_results, 1):
            meta = result.get("meta", {})
            score = result.get("score", 0)
            doc_id = result.get("id", f"doc_{i}")
            
            # Aggiungi a tracking
            retrieved_docs.append({
                "id": doc_id,
                "score": score,
                "title": meta.get("title", "N/A"),
                "content": meta.get("content", "")[:500]  # Primi 500 caratteri
            })
            
            context_parts.append(f"[DOCUMENTO {i}] ID: {doc_id} (Score: {score:.4f})")
            
            if "title" in meta:
                context_parts.append(f"Titolo: {meta['title']}")
            if "author" in meta:
                context_parts.append(f"Autore: {meta['author']}")
            if "category" in meta:
                context_parts.append(f"Categoria: {meta['category']}")
                
            if "content" in meta:
                context_parts.append(f"Contenuto: {meta['content']}")
                
            context_parts.append("=" * 50)
            
        # Aggiungi al context manager
        context_summary = context_manager.get_context_summary()
        if context_summary and context_summary != "Nessun contesto conversazionale precedente.":
            context_parts.append(f"\\n=== CONTESTO CONVERSAZIONALE PRECEDENTE ===")
            context_parts.append(context_summary)
            
        # Salva documenti recuperati per tracking
        context_manager.retrieved_contexts = retrieved_docs
        
        return "\\n".join(context_parts)
        
    except Exception as e:
        error_msg = f"Errore nella creazione contesto RAG: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def update_conversation_history(user_query: str, response: str) -> str:
    """
    Aggiorna la cronologia conversazionale del sistema RAG.
    
    Args:
        user_query: Query dell'utente
        response: Risposta generata
        
    Returns:
        str: Conferma aggiornamento cronologia
    """
    global context_manager
    
    try:
        # Recupera ultimi documenti utilizzati
        retrieved_docs = getattr(context_manager, 'retrieved_contexts', [])
        
        # Aggiungi turno conversazionale
        context_manager.add_turn(user_query, retrieved_docs, response)
        
        return f"Cronologia conversazionale aggiornata. Turni totali: {len(context_manager.conversation_history)}"
        
    except Exception as e:
        error_msg = f"Errore nell'aggiornamento cronologia: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def get_database_stats() -> str:
    """
    Ottiene statistiche del database vettoriale.
    
    Returns:
        str: Statistiche formattate del database
    """
    global vector_db
    
    if not vector_db:
        return "Errore: Database vettoriale non inizializzato. Usa 'initialize_vector_database' prima."
        
    try:
        stats = vector_db.get_stats()
        
        formatted_stats = []
        formatted_stats.append("=== STATISTICHE DATABASE VETTORIALE ===")
        formatted_stats.append(f"Record totali: {stats['total_records']}")
        formatted_stats.append(f"Dimensioni vettori: {stats['dimension']}")
        formatted_stats.append(f"Metrica similarit�: {stats['metric']}")
        formatted_stats.append(f"Campi configurati: {', '.join(stats['fields'])}")
        formatted_stats.append(f"Modello embedding: {stats['embedding_model']}")
        formatted_stats.append(f"Indice FAISS size: {stats['index_size']}")
        
        # Esempi di record
        record_ids = vector_db.list_records(limit=5)
        if record_ids:
            formatted_stats.append(f"\\nEsempi ID record: {', '.join(record_ids)}")
            
        return "\\n".join(formatted_stats)
        
    except Exception as e:
        error_msg = f"Errore nel recupero statistiche: {str(e)}"
        logger.error(error_msg)
        return error_msg


def build_llm_model(model_id: str, local_model_path: str = None):
    """Costruisce il modello LLM per l'agente RAG."""
    try:
        if local_model_path:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForCausalLM.from_pretrained(local_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)

        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, return_full_text=False)
        return pipe
    except Exception as e:
        logger.error(f"Errore nel caricamento modello: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Sistema RAG con HuggingFace Agents e Database Vettoriale FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:

1. Query RAG semplice:
   python LLMrag.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Cos'� il machine learning?"

2. RAG con database personalizzato:
   python LLMrag.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Spiega le reti neurali" --db_path "./my_knowledge_base"

3. Query multi-step avanzata:
   python LLMrag.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --query "Trova informazioni su Python e poi confronta con TensorFlow"

4. Modalit� interattiva:
   python LLMrag.py --model "microsoft/DialoGPT-medium" --local_model_path "/path/to/model" --interactive

STRUMENTI RAG DISPONIBILI:
  " initialize_vector_database: Carica database vettoriale FAISS
  " semantic_search: Ricerca semantica su contenuti
  " keyword_search: Ricerca per parole chiave
  " get_document_by_id: Recupera documento specifico
  " create_rag_context: Crea contesto per generazione RAG
  " update_conversation_history: Gestisce cronologia conversazionale
  " get_database_stats: Statistiche database

PIPELINE RAG:
1. Inizializza database vettoriale (FAISS)
2. Ricerca documenti rilevanti (semantic/keyword)
3. Crea contesto aumentato
4. Genera risposta con contesto
5. Aggiorna cronologia conversazionale
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="Nome del modello HuggingFace per l'agente RAG")
    parser.add_argument("--query", type=str,
                       help="Query per il sistema RAG")
    parser.add_argument("--local_model_path", type=str, default=None,
                       help="Path al modello locale (sovrascrive --model se fornito)")
    parser.add_argument("--db_path", type=str, default="./vector_db",
                       help="Path al database vettoriale FAISS")
    parser.add_argument("--max_docs", type=int, default=3,
                       help="Numero massimo di documenti da recuperare per RAG")
    parser.add_argument("--similarity_threshold", type=float, default=0.3,
                       help="Soglia di similarit� per filtrare documenti")
    parser.add_argument("--interactive", action="store_true",
                       help="Modalit� interattiva per query multiple")
    parser.add_argument("--verbose", action="store_true",
                       help="Abilita logging verboso")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Aggiorna configurazione RAG
    global rag_config
    rag_config.vector_db_path = args.db_path
    rag_config.max_retrieved_docs = args.max_docs
    rag_config.similarity_threshold = args.similarity_threshold
    
    logger.info("=== INIZIALIZZAZIONE SISTEMA RAG ===")
    logger.info(f"Modello: {args.local_model_path or args.model}")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Max documenti: {args.max_docs}")
    logger.info(f"Soglia similarit�: {args.similarity_threshold}")
    
    # Carica modello LLM
    try:
        model = TransformersModel(model_id=args.local_model_path or args.model)
    except Exception as e:
        logger.error(f"Errore caricamento modello: {e}")
        return
    
    # Crea agente RAG con tools specializzati
    rag_tools = [
        initialize_vector_database,
        semantic_search,
        keyword_search,
        get_document_by_id,
        create_rag_context,
        update_conversation_history,
        get_database_stats
    ]
    
    agent = CodeAgent(tools=rag_tools, model=model)
    
    logger.info("=== AGENTE RAG INIZIALIZZATO ===")
    
    # Inizializza automaticamente il database
    init_result = agent.run(f"initialize_vector_database('{args.db_path}')")
    print(f"\\n=' INIZIALIZZAZIONE DATABASE:")
    print(init_result)
    
    if args.interactive:
        # Modalit� interattiva
        print("\\n" + "="*60)
        print("MODALIT� INTERATTIVA SISTEMA RAG")
        print("Digita 'exit' per uscire, 'stats' per statistiche database")
        print("="*60 + "\\n")
        
        while True:
            try:
                user_query = input("\\n> RAG Query: ").strip()
                
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    print("\\n=K Arrivederci!")
                    break
                    
                if user_query.lower() == 'stats':
                    stats_result = agent.run("get_database_stats()")
                    print(f"\\n=� STATISTICHE DATABASE:\\n{stats_result}")
                    continue
                    
                if not user_query:
                    continue
                
                # Costruisci prompt RAG completo
                rag_prompt = f\"\"\"
Sei un assistente AI specializzato in Retrieval-Augmented Generation (RAG).

Per rispondere alla query dell'utente, segui questa pipeline:
1. Usa 'create_rag_context' per recuperare documenti rilevanti
2. Analizza il contesto recuperato
3. Genera una risposta dettagliata basata sul contesto
4. Usa 'update_conversation_history' per aggiornare la cronologia

Query utente: {user_query}

Fornisci una risposta completa e accurata basata sui documenti recuperati.
\"\"\"
                
                print(f"\\n=
 Elaborazione query: {user_query}")
                print("� Ricerca documenti e generazione risposta...")
                
                # Esegui pipeline RAG
                result = agent.run(rag_prompt)
                
                print(f"\\n<� RISPOSTA RAG:")
                print("="*50)
                print(result)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\\n\\n=K Sessione terminata dall'utente.")
                break
            except Exception as e:
                print(f"\\nL Errore: {str(e)}")
                
    elif args.query:
        # Singola query
        print(f"\\n=
 PROCESSING QUERY: {args.query}")
        print("="*60)
        
        # Costruisci prompt RAG
        rag_prompt = f\"\"\"
Sei un assistente AI specializzato in Retrieval-Augmented Generation (RAG).

Per rispondere alla query dell'utente, segui questa pipeline:
1. Usa 'create_rag_context' per recuperare documenti rilevanti dalla knowledge base
2. Analizza attentamente il contesto recuperato
3. Genera una risposta dettagliata e accurata basata sui documenti trovati
4. Cita le fonti quando possibile
5. Usa 'update_conversation_history' per salvare la conversazione

Query utente: {args.query}

Fornisci una risposta completa, accurata e ben strutturata.
\"\"\"
        
        # Esegui pipeline RAG
        result = agent.run(rag_prompt)
        
        print(f"\\n<� RISPOSTA RAG:")
        print("="*60)
        print(result)
        print("="*60)
        
    else:
        # Mostra help se nessuna query
        print("\\nL Nessuna query fornita.")
        print("Usa --query 'tua domanda' per una singola query")
        print("Usa --interactive per modalit� interattiva")
        print("Usa --help per vedere tutte le opzioni")


if __name__ == "__main__":
    main()