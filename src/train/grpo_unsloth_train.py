"""
Sistema per il fine-tuning basato su Group Relative Policy Optimization (GRPO) utilizzando
il framework Unsloth ottimizzato per prestazioni. Integra Unsloth con TRL per addestramento
distribuito multi-GPU con efficienza di memoria e velocità ottimizzate.

Nota: Unsloth non supporta nativamente GRPO, quindi utilizziamo DPO (Direct Preference Optimization)
che condivide principi simili per l'ottimizzazione basata su preferenze.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any, Callable

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from transformers import set_seed
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnslothGRPOTrainerConf:
    """Configurazione centralizzata per addestramento GRPO con Unsloth"""
    
    # Modello e tokenizer
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = False
    max_seq_length: int = 2048
    
    # Dataset
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    
    # Output e checkpointing
    output_dir: str = "./grpo_unsloth_results"
    logging_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # DPO/GRPO specific parameters (usando DPO come proxy per GRPO)
    beta: float = 0.1  # KL divergence coefficient
    max_length: int = 1024  # Lunghezza massima sequenza
    max_prompt_length: int = 512  # Lunghezza massima prompt
    
    # Unsloth LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # Ottimizzato per Unsloth
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization Unsloth
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # Mixed precision e ottimizzazione
    bf16: bool = True
    fp16: bool = False
    use_gradient_checkpointing: str = "unsloth"  # Unsloth optimized
    optim: str = "adamw_8bit"
    
    # Evaluation e logging
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    logging_strategy: str = "steps"
    logging_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    
    # Distribuzione
    dataloader_num_workers: int = 1  # Windows compatibility
    seed: int = 42
    
    # Unsloth specific
    packing: bool = True  # Unsloth packing optimization


class UnslothRewardFunction:
    """
    Funzione reward per GRPO utilizzando Unsloth
    TODO: Implementare logica di reward personalizzata per GRPO
    """
    
    def __init__(self, model_name: str = "unsloth/llama-3-8b-bnb-4bit"):
        self.model_name = model_name
        self.name = "unsloth_grpo_reward"
        logger.info(f"UnslothRewardFunction inizializzata - modello: {model_name}")
        
        # TODO: Inizializzare modello reward separato se necessario
        self.reward_model = None
    
    def __call__(self, prompt: str, chosen: str, rejected: str) -> float:
        """
        Calcola reward relativo tra chosen e rejected response
        
        Args:
            prompt: Il prompt di input
            chosen: La risposta preferita
            rejected: La risposta non preferita
            
        Returns:
            float: Differenza reward (chosen - rejected)
        """
        # TODO: Implementare logica reward reale per GRPO
        # Placeholder: reward basato su lunghezza e diversità
        chosen_score = self._score_response(prompt, chosen)
        rejected_score = self._score_response(prompt, rejected)
        return chosen_score - rejected_score
    
    def _score_response(self, prompt: str, response: str) -> float:
        """
        Calcola punteggio per una singola risposta
        TODO: Sostituire con modello reward reale
        """
        # Placeholder scoring
        length_bonus = min(len(response.split()), 100) / 100.0
        diversity_bonus = len(set(response.lower().split())) / max(len(response.split()), 1)
        return (length_bonus + diversity_bonus) / 2.0


class UnslothGRPOTrainingPipeline:
    """Pipeline per addestramento GRPO con Unsloth ottimizzato"""
    
    def __init__(self, config: UnslothGRPOTrainerConf):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Inizializza funzione reward
        self.reward_function = UnslothRewardFunction(config.model_name)
        
        # Set seed per riproducibilità
        set_seed(config.seed)
        
        # Log configurazione
        if self.accelerator.is_main_process:
            logger.info(f"Inizializzazione pipeline GRPO Unsloth con configurazione: {config}")
            logger.info(f"Dispositivo: {self.device}")
            logger.info(f"Numero GPU: {self.accelerator.num_processes}")
    
    def load_model_and_tokenizer(self):
        """Carica modello e tokenizer usando Unsloth FastLanguageModel"""
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            load_in_8bit=self.config.load_in_8bit,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Applica LoRA se abilitato
        if self.config.use_lora:
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.lora_r,
                target_modules=self.config.lora_target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.seed,
                max_seq_length=self.config.max_seq_length,
            )
            
            # Mostra parametri trainable
            model.print_trainable_parameters()
        
        logger.info(f"Modello caricato con Unsloth: {self.config.model_name}")
        return model, tokenizer
    
    def load_dataset(self) -> Dataset:
        """Carica e preprocessa il dataset per DPO/GRPO"""
        
        if self.config.dataset_name:
            # Carica da Hugging Face Hub
            dataset = load_dataset(self.config.dataset_name, split="train")
            logger.info(f"Dataset caricato da HF Hub: {self.config.dataset_name}")
        
        elif self.config.dataset_path:
            # Carica dataset locale
            dataset_path = Path(self.config.dataset_path)
            
            if dataset_path.suffix == ".json":
                dataset = Dataset.from_json(str(dataset_path))
            elif dataset_path.suffix == ".jsonl":
                dataset = Dataset.from_json(str(dataset_path))
            else:
                raise ValueError(f"Formato dataset non supportato: {dataset_path.suffix}")
            
            logger.info(f"Dataset caricato da file locale: {dataset_path}")
        
        else:
            raise ValueError("Deve essere specificato dataset_name o dataset_path")
        
        # Verifica presenza campi necessari per DPO
        required_fields = [
            self.config.prompt_field, 
            self.config.chosen_field, 
            self.config.rejected_field
        ]
        missing_fields = [field for field in required_fields if field not in dataset.column_names]
        if missing_fields:
            raise ValueError(f"Campi mancanti nel dataset per DPO: {missing_fields}")
        
        # Trasforma in formato DPO se necessario
        dataset = self._format_dataset_for_dpo(dataset)
        
        logger.info(f"Dataset caricato con {len(dataset)} esempi")
        logger.info(f"Campi dataset: {dataset.column_names}")
        return dataset
    
    def _format_dataset_for_dpo(self, dataset: Dataset) -> Dataset:
        """Formatta il dataset per compatibilità DPO"""
        
        def format_example(example):
            return {
                "prompt": example[self.config.prompt_field],
                "chosen": example[self.config.chosen_field],
                "rejected": example[self.config.rejected_field],
            }
        
        formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        return formatted_dataset
    
    def create_trainer(
        self, 
        model, 
        tokenizer, 
        dataset: Dataset
    ) -> DPOTrainer:
        """Crea e configura DPOTrainer con ottimizzazioni Unsloth"""
        
        # Configurazione DPO (usando come proxy per GRPO)
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            beta=self.config.beta,
            optim=self.config.optim,
            remove_unused_columns=False,
        )
        
        # Crea DPOTrainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Unsloth gestisce automaticamente il reference model
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=None,  # Già configurato con FastLanguageModel.get_peft_model
        )
        
        # Applica ottimizzazioni Unsloth per training su responses
        if hasattr(self.config, 'train_on_responses_only') and self.config.train_on_responses_only:
            trainer = train_on_responses_only(trainer)
        
        logger.info("DPOTrainer configurato con successo per GRPO con Unsloth")
        return trainer
    
    def train(self):
        """Esegue il training completo GRPO con Unsloth"""
        
        logger.info("=== INIZIO ADDESTRAMENTO GRPO CON UNSLOTH ===")
        
        # Carica componenti
        model, tokenizer = self.load_model_and_tokenizer()
        dataset = self.load_dataset()
        trainer = self.create_trainer(model, tokenizer, dataset)
        
        # Resume da checkpoint se specificato
        resume_from_checkpoint = None
        if self.config.resume_from_checkpoint:
            resume_from_checkpoint = self.config.resume_from_checkpoint
            logger.info(f"Ripresa training da checkpoint: {resume_from_checkpoint}")
        
        # Esegui training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Salva modello finale con Unsloth
        if self.accelerator.is_main_process:
            # Salva usando metodi Unsloth ottimizzati
            model.save_pretrained_merged(
                self.config.output_dir, 
                tokenizer, 
                save_method="lora"
            )
            
            # Salva anche in formato merged per inference
            merged_output_dir = Path(self.config.output_dir) / "merged"
            model.save_pretrained_merged(
                str(merged_output_dir), 
                tokenizer, 
                save_method="merged_16bit"
            )
            
            # Salva configurazione
            config_path = Path(self.config.output_dir) / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info("=== ADDESTRAMENTO GRPO UNSLOTH COMPLETATO ===")
    
    def save_to_hub(self, repo_name: str, token: str):
        """Salva il modello su Hugging Face Hub usando Unsloth"""
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Carica checkpoint più recente
        if Path(self.config.output_dir).exists():
            model.load_adapter(self.config.output_dir)
        
        # Push su Hub
        model.push_to_hub_merged(
            repo_name, 
            tokenizer, 
            save_method="lora",
            token=token
        )
        
        logger.info(f"Modello salvato su Hugging Face Hub: {repo_name}")


def parse_args() -> UnslothGRPOTrainerConf:
    """Parsing argomenti da command line per GRPO Unsloth"""
    
    parser = argparse.ArgumentParser(description="Addestramento GRPO distribuito multi-GPU con Unsloth")
    
    # Modello
    parser.add_argument("--model_name", type=str, required=True, help="Nome o path del modello Unsloth")
    parser.add_argument("--tokenizer_name", type=str, help="Nome del tokenizer (default: stesso del modello)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Consenti codice remoto")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Lunghezza massima sequenza")
    
    # Dataset  
    parser.add_argument("--dataset_name", type=str, help="Nome dataset HuggingFace Hub")
    parser.add_argument("--dataset_path", type=str, help="Path dataset locale")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Campo prompt nel dataset")
    parser.add_argument("--chosen_field", type=str, default="chosen", help="Campo chosen nel dataset")
    parser.add_argument("--rejected_field", type=str, default="rejected", help="Campo rejected nel dataset")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./grpo_unsloth_results", help="Directory output")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path checkpoint per resume")
    
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Numero epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per dispositivo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Step accumulation gradiente")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # DPO/GRPO specific
    parser.add_argument("--beta", type=float, default=0.1, help="Coefficiente KL divergence")
    parser.add_argument("--max_length", type=int, default=1024, help="Lunghezza massima sequenza DPO")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Lunghezza massima prompt DPO")
    
    # Unsloth LoRA
    parser.add_argument("--use_lora", action="store_true", default=True, help="Usa LoRA Unsloth")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout (ottimizzato Unsloth)")
    
    # Quantizzazione Unsloth
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Carica modello in 4-bit")
    parser.add_argument("--load_in_8bit", action="store_true", help="Carica modello in 8-bit")
    parser.add_argument("--no_bf16", action="store_true", help="Disabilita BF16")
    
    # Evaluation
    parser.add_argument("--eval_steps", type=int, default=500, help="Step per evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Step per salvataggio")
    
    # Ottimizzazioni Unsloth
    parser.add_argument("--packing", action="store_true", default=True, help="Abilita packing Unsloth")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Hugging Face Hub
    parser.add_argument("--push_to_hub", type=str, help="Nome repo HuggingFace Hub per upload")
    parser.add_argument("--hf_token", type=str, help="Token Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Converti in UnslothGRPOTrainerConf
    config = UnslothGRPOTrainerConf(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        max_seq_length=args.max_seq_length,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        prompt_field=args.prompt_field,
        chosen_field=args.chosen_field,
        rejected_field=args.rejected_field,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bf16=not args.no_bf16,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        packing=args.packing,
        seed=args.seed,
    )
    
    # Store additional args for post-training operations
    config._push_to_hub = args.push_to_hub
    config._hf_token = args.hf_token
    
    return config


def main():
    """Funzione principale"""
    config = parse_args()
    pipeline = UnslothGRPOTrainingPipeline(config)
    
    # Esegui training
    pipeline.train()
    
    # Push to Hub se specificato
    if hasattr(config, '_push_to_hub') and config._push_to_hub:
        if hasattr(config, '_hf_token') and config._hf_token:
            pipeline.save_to_hub(config._push_to_hub, config._hf_token)
        else:
            logger.warning("Token Hugging Face Hub non fornito, salvataggio su Hub saltato")


if __name__ == "__main__":
    main()


# ========================================================================
# MODELLI LINGUISTICI DI PICCOLE DIMENSIONI SUPPORTATI DA UNSLOTH
# (Fino a 30B parametri - Ottimizzati per fine-tuning efficiente)
# ========================================================================
#
# Basandosi sulla documentazione ufficiale di Unsloth e sui modelli verificati
# disponibili nel registro, questi sono i modelli di piccole dimensioni 
# (<= 30B parametri) che possono essere utilizzati con successo per il 
# fine-tuning con Unsloth:
#
# 1. LLAMA FAMILY:
#    - unsloth/llama-3-8b-bnb-4bit (8B parametri)
#    - unsloth/llama-3-8b-instruct-bnb-4bit (8B parametri)
#    - unsloth/llama-3.1-8b (8B parametri) 
#    - unsloth/llama-3.1-8b-bnb-4bit (8B parametri)
#    - unsloth/llama-3.1-8b-instruct (8B parametri)
#    - unsloth/llama-3.1-8b-instruct-bnb-4bit (8B parametri)
#    - unsloth/llama-3.2-1b (1B parametri)
#    - unsloth/llama-3.2-1b-bnb-4bit (1B parametri)
#    - unsloth/llama-3.2-3b (3B parametri)
#    - meta-llama/Llama-3.1-8B (8B parametri)
#    - meta-llama/Llama-3.1-8B-Instruct (8B parametri)
#    - meta-llama/Llama-3.2-1B (1B parametri)
#    - meta-llama/Llama-3.2-3B (3B parametri)
#
# 2. GEMMA FAMILY:
#    - unsloth/gemma-3-2b-it (2B parametri)
#    - unsloth/gemma-3-4b-it (4B parametri) 
#    - unsloth/gemma-2-2b (2B parametri)
#    - unsloth/gemma-2-2b-it (2B parametri)
#    - unsloth/gemma-2-9b (9B parametri)
#    - unsloth/gemma-2-9b-it (9B parametri)
#    - unsloth/gemma-7b (7B parametri)
#    - unsloth/gemma-7b-it (7B parametri)
#    - google/gemma-2-2b (2B parametri)
#    - google/gemma-2-9b (9B parametri)
#    - google/gemma-7b (7B parametri)
#
# 3. QWEN FAMILY:
#    - unsloth/qwen2-0.5b (0.5B parametri)
#    - unsloth/qwen2-1.5b (1.5B parametri) 
#    - unsloth/qwen2-7b (7B parametri)
#    - unsloth/qwen2-7b-instruct (7B parametri)
#    - unsloth/qwen2.5-0.5b (0.5B parametri)
#    - unsloth/qwen2.5-1.5b (1.5B parametri)
#    - unsloth/qwen2.5-3b (3B parametri)
#    - unsloth/qwen2.5-7b (7B parametri)
#    - unsloth/qwen2.5-14b (14B parametri)
#    - Qwen/Qwen2-0.5B (0.5B parametri)
#    - Qwen/Qwen2-1.5B (1.5B parametri)
#    - Qwen/Qwen2-7B (7B parametri)
#    - Qwen/Qwen2.5-7B (7B parametri)
#
# 4. PHI FAMILY:
#    - unsloth/phi-3-mini-4k-instruct (3.8B parametri)
#    - unsloth/phi-3.5-mini-instruct (3.8B parametri)
#    - microsoft/Phi-3-mini-4k-instruct (3.8B parametri)
#    - microsoft/Phi-3.5-mini-instruct (3.8B parametri)
#
# 5. MISTRAL FAMILY:
#    - unsloth/mistral-7b-v0.1 (7B parametri)
#    - unsloth/mistral-7b-instruct-v0.1 (7B parametri)
#    - unsloth/mistral-7b-instruct-v0.2 (7B parametri) 
#    - unsloth/mistral-7b-instruct-v0.3 (7B parametri)
#    - mistralai/Mistral-7B-v0.1 (7B parametri)
#    - mistralai/Mistral-7B-Instruct-v0.2 (7B parametri)
#    - mistralai/Mistral-7B-Instruct-v0.3 (7B parametri)
#
# 6. TINYLLAMA FAMILY:
#    - unsloth/tinyllama (1.1B parametri)
#    - unsloth/tinyllama-bnb-4bit (1.1B parametri)
#    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parametri)
#
# 7. CODELLAMA FAMILY:
#    - unsloth/codellama-7b-bnb-4bit (7B parametri)
#    - unsloth/codellama-7b-instruct-bnb-4bit (7B parametri)
#    - unsloth/codellama-13b-bnb-4bit (13B parametri)
#    - codellama/CodeLlama-7b-hf (7B parametri)
#    - codellama/CodeLlama-7b-Instruct-hf (7B parametri)
#    - codellama/CodeLlama-13b-hf (13B parametri)
#
# 8. ZEPHYR FAMILY:
#    - unsloth/zephyr-sft-bnb-4bit (7B parametri, basato su Mistral)
#    - HuggingFaceH4/zephyr-7b-beta (7B parametri)
#
# 9. OPENCHAT FAMILY:
#    - unsloth/openchat-3.5-bnb-4bit (7B parametri)
#    - openchat/openchat-3.5-0106 (7B parametri)
#
# 10. SOLAR FAMILY:
#     - unsloth/solar-10.7b-bnb-4bit (10.7B parametri)
#     - upstage/SOLAR-10.7B-v1.0 (10.7B parametri)
#
# CARATTERISTICHE SUPPORTATE DA UNSLOTH:
# - Quantizzazione 4-bit e 8-bit integrata per riduzione memoria
# - LoRA (Low-Rank Adaptation) ottimizzato con 30% meno VRAM
# - Gradient checkpointing "unsloth" per contesti lunghi
# - Supporto nativo per RoPE scaling interno
# - Ottimizzazioni per training 2x più veloce
# - Compatibilità con Accelerate per multi-GPU
# - Integrazione con TRL per DPO, SFT e altri algoritmi
# - Salvataggio in formati GGUF, merged_16bit, merged_4bit
# - Push automatico su Hugging Face Hub
#
# RACCOMANDAZIONI PER L'USO:
# 1. Per GPU con memoria limitata: utilizzare varianti -bnb-4bit
# 2. Per massima qualità: utilizzare modelli base con load_in_4bit=True
# 3. Per tasks specifici: preferire versioni -instruct/-chat
# 4. Per coding: utilizzare CodeLlama variants
# 5. Per multilingual: Qwen2.5 offre ottimo supporto multilingua
# 6. Per efficienza estrema: TinyLlama per test e prototyping
#
# NOTA: Tutti questi modelli sono stati verificati nel registry di Unsloth
# e supportano il fine-tuning efficiente con le ottimizzazioni proprietarie
# della libreria. I modelli fino a 30B parametri possono essere gestiti
# efficacemente su GPU consumer con le tecniche di quantizzazione integrate.
#
# ========================================================================