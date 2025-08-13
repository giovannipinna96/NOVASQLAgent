"""
Sistema per il fine-tuning supervisionato (SFT) di modelli linguistici compatibili con Hugging Face,
utilizzando le librerie ufficiali Hugging Face come `SFTTrainer`, `PEFT` e `LoRA`. Ottimizzato per
addestramento distribuito su multiple GPU all'interno dello stesso nodo usando la libreria `Accelerate`.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any

import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainerConf:
    """Configurazione centralizzata per tutti i parametri di addestramento"""
    
    # Modello e tokenizer
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = False
    
    # Dataset
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    text_field: str = "text"
    max_seq_length: int = 1024
    
    # Output e checkpointing
    output_dir: str = "./results"
    logging_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Mixed precision e ottimizzazione
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Evaluation e logging
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    logging_strategy: str = "steps"
    logging_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    
    # Distribuzione
    ddp_find_unused_parameters: bool = False
    seed: int = 42


class SFTTrainingPipeline:
    """Pipeline per addestramento SFT con supporto distribuito multi-GPU"""
    
    def __init__(self, config: TrainerConf):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Set seed per riproducibilità
        set_seed(config.seed)
        
        # Log configurazione
        if self.accelerator.is_main_process:
            logger.info(f"Inizializzazione pipeline SFT con configurazione: {config}")
            logger.info(f"Dispositivo: {self.device}")
            logger.info(f"Numero GPU: {self.accelerator.num_processes}")
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Carica e configura il tokenizer"""
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # Aggiungi pad token se non presente
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Tokenizer caricato: {tokenizer_name}")
        return tokenizer
    
    def load_model(self, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
        """Carica e configura il modello con quantizzazione e LoRA"""
        
        # Configurazione quantizzazione 4-bit
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            )
        
        # Carica modello base
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.config.use_4bit else None,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )
        
        # Ridimensiona embeddings se necessario
        model.resize_token_embeddings(len(tokenizer))
        
        # Prepara per training con quantizzazione
        if self.config.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configura LoRA se abilitato
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        logger.info(f"Modello caricato: {self.config.model_name}")
        return model
    
    def load_dataset(self) -> Dataset:
        """Carica e preprocessa il dataset"""
        
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
        
        logger.info(f"Dataset caricato con {len(dataset)} esempi")
        return dataset
    
    def create_trainer(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        dataset: Dataset
    ) -> SFTTrainer:
        """Crea e configura SFTTrainer"""
        
        # Configurazione SFT
        sft_config = SFTConfig(
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
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            logging_strategy=self.config.logging_strategy,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
            seed=self.config.seed,
            max_seq_length=self.config.max_seq_length,
            remove_unused_columns=False,
        )
        
        # Crea SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            dataset_text_field=self.config.text_field,
        )
        
        logger.info("SFTTrainer configurato con successo")
        return trainer
    
    def train(self):
        """Esegue il training completo"""
        
        logger.info("=== INIZIO ADDESTRAMENTO SFT ===")
        
        # Carica componenti
        tokenizer = self.load_tokenizer()
        model = self.load_model(tokenizer)
        dataset = self.load_dataset()
        trainer = self.create_trainer(model, tokenizer, dataset)
        
        # Resume da checkpoint se specificato
        resume_from_checkpoint = None
        if self.config.resume_from_checkpoint:
            resume_from_checkpoint = self.config.resume_from_checkpoint
            logger.info(f"Ripresa training da checkpoint: {resume_from_checkpoint}")
        
        # Esegui training
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Salva modello finale
        if self.accelerator.is_main_process:
            trainer.save_model()
            tokenizer.save_pretrained(self.config.output_dir)
            
            # Salva configurazione
            config_path = Path(self.config.output_dir) / "training_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info("=== ADDESTRAMENTO COMPLETATO ===")


def parse_args() -> TrainerConf:
    """Parsing argomenti da command line"""
    
    parser = argparse.ArgumentParser(description="Addestramento SFT distribuito multi-GPU")
    
    # Modello
    parser.add_argument("--model_name", type=str, required=True, help="Nome o path del modello")
    parser.add_argument("--tokenizer_name", type=str, help="Nome del tokenizer (default: stesso del modello)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Consenti codice remoto")
    
    # Dataset  
    parser.add_argument("--dataset_name", type=str, help="Nome dataset HuggingFace Hub")
    parser.add_argument("--dataset_path", type=str, help="Path dataset locale")
    parser.add_argument("--text_field", type=str, default="text", help="Campo testo nel dataset")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Lunghezza massima sequenza")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./sft_results", help="Directory output")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path checkpoint per resume")
    
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Numero epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per dispositivo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Step accumulation gradiente")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True, help="Usa LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Quantizzazione
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Usa quantizzazione 4-bit")
    parser.add_argument("--no_bf16", action="store_true", help="Disabilita BF16")
    
    # Evaluation
    parser.add_argument("--eval_steps", type=int, default=500, help="Step per evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Step per salvataggio")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Converti in TrainerConf
    config = TrainerConf(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        bf16=not args.no_bf16,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )
    
    return config


def main():
    """Funzione principale"""
    config = parse_args()
    pipeline = SFTTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()