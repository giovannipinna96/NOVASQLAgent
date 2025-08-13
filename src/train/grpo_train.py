"""
Sistema per il fine-tuning basato su Group Relative Policy Optimization (GRPO) di modelli linguistici 
compatibili con Hugging Face. Il codice integra `GRPOTrainer`, usa LoRA/PEFT per aggiornamenti 
parametrici efficienti e utilizza la libreria `Accelerate` per addestramento distribuito multi-GPU.
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
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOTrainerConf:
    """Configurazione centralizzata per tutti i parametri di addestramento GRPO"""
    
    # Modello e tokenizer
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: Optional[str] = None
    trust_remote_code: bool = False
    
    # Dataset
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    
    # Output e checkpointing
    output_dir: str = "./grpo_results"
    logging_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # GRPO specific parameters
    beta: float = 0.1  # KL divergence coefficient
    group_size: int = 8  # Dimensione gruppo per confronti relativi
    response_length: int = 256  # Lunghezza massima risposta generata
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
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


class RewardFunction:
    """
    Funzione reward placeholder per GRPO
    TODO: Implementare logica di reward personalizzata
    """
    
    def __init__(self):
        self.name = "placeholder_reward"
        logger.info("RewardFunction inizializzata - implementazione placeholder")
    
    def __call__(self, prompt: str, response: str) -> float:
        """
        Calcola reward per una coppia prompt-response
        
        Args:
            prompt: Il prompt di input
            response: La risposta generata
            
        Returns:
            float: Punteggio reward (placeholder)
        """
        # TODO: Implementare logica reward reale
        # Placeholder: reward basato su lunghezza
        length_bonus = min(len(response.split()), 50) / 50.0
        return length_bonus
    
    def batch_reward(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Calcola rewards per un batch di prompt-response
        
        Args:
            prompts: Lista di prompts
            responses: Lista di risposte
            
        Returns:
            List[float]: Lista di rewards
        """
        return [self(prompt, response) for prompt, response in zip(prompts, responses)]


class GRPOTrainingPipeline:
    """Pipeline per addestramento GRPO con supporto distribuito multi-GPU"""
    
    def __init__(self, config: GRPOTrainerConf):
        self.config = config
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        # Inizializza funzione reward
        self.reward_function = RewardFunction()
        
        # Set seed per riproducibilità
        set_seed(config.seed)
        
        # Log configurazione
        if self.accelerator.is_main_process:
            logger.info(f"Inizializzazione pipeline GRPO con configurazione: {config}")
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
        """Carica e preprocessa il dataset per GRPO"""
        
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
        
        # Verifica presenza campi necessari
        required_fields = [self.config.prompt_field, self.config.completion_field]
        missing_fields = [field for field in required_fields if field not in dataset.column_names]
        if missing_fields:
            raise ValueError(f"Campi mancanti nel dataset: {missing_fields}")
        
        logger.info(f"Dataset caricato con {len(dataset)} esempi")
        logger.info(f"Campi dataset: {dataset.column_names}")
        return dataset
    
    def create_trainer(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        dataset: Dataset
    ) -> GRPOTrainer:
        """Crea e configura GRPOTrainer"""
        
        # Configurazione GRPO
        grpo_config = GRPOConfig(
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
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_prompt_length,
            beta=self.config.beta,
            group_size=self.config.group_size,
            response_length=self.config.response_length,
            remove_unused_columns=False,
        )
        
        # Crea GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            reward_function=self.reward_function.batch_reward,
        )
        
        logger.info("GRPOTrainer configurato con successo")
        return trainer
    
    def train(self):
        """Esegue il training completo GRPO"""
        
        logger.info("=== INIZIO ADDESTRAMENTO GRPO ===")
        
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
        
        logger.info("=== ADDESTRAMENTO GRPO COMPLETATO ===")


def parse_args() -> GRPOTrainerConf:
    """Parsing argomenti da command line per GRPO"""
    
    parser = argparse.ArgumentParser(description="Addestramento GRPO distribuito multi-GPU")
    
    # Modello
    parser.add_argument("--model_name", type=str, required=True, help="Nome o path del modello")
    parser.add_argument("--tokenizer_name", type=str, help="Nome del tokenizer (default: stesso del modello)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Consenti codice remoto")
    
    # Dataset  
    parser.add_argument("--dataset_name", type=str, help="Nome dataset HuggingFace Hub")
    parser.add_argument("--dataset_path", type=str, help="Path dataset locale")
    parser.add_argument("--prompt_field", type=str, default="prompt", help="Campo prompt nel dataset")
    parser.add_argument("--completion_field", type=str, default="completion", help="Campo completion nel dataset")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Lunghezza massima sequenza")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Lunghezza massima prompt")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./grpo_results", help="Directory output")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path checkpoint per resume")
    
    # Training
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Numero epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per dispositivo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Step accumulation gradiente")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # GRPO specific
    parser.add_argument("--beta", type=float, default=0.1, help="Coefficiente KL divergence")
    parser.add_argument("--group_size", type=int, default=8, help="Dimensione gruppo per confronti")
    parser.add_argument("--response_length", type=int, default=256, help="Lunghezza massima risposta")
    
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
    
    # Converti in GRPOTrainerConf
    config = GRPOTrainerConf(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        group_size=args.group_size,
        response_length=args.response_length,
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
    pipeline = GRPOTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()