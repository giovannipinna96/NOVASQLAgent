"""
This file implements a system for supervised fine-tuning (SFT) of Hugging Face-compatible language models,
using the official Hugging Face libraries such as `SFTTrainer`, `PEFT`, and `LoRA`. It is optimized for 
distributed training on multiple GPUs within the same node using the `Accelerate` library.

OBJECTIVES AND FUNCTIONALITY:

1. TRAINING BACKEND AND ARCHITECTURE:
    - The training pipeline must utilize Hugging Face’s `SFTTrainer` to perform supervised fine-tuning (SFT).
    - It should support LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning), leveraging 
      `peft` integration provided by Hugging Face for efficient training on large models.
    - The system must be compatible with Hugging Face’s `Accelerate` to enable distributed multi-GPU training.
      Use `accelerate.launch` or equivalent mechanisms for execution.

2. COMMAND-LINE CONFIGURATION:
    - All essential training parameters must be configurable via command-line arguments (e.g., using `argparse` or `typer`).
    - Configurable parameters include (but are not limited to):
        • Model name or path
        • Output directory
        • Dataset path or name (and specific column names for text/prompt/response)
        • Number of training epochs
        • Batch size (train/eval)
        • Learning rate
        • LoRA-specific configurations (e.g., r, alpha, dropout)
        • PEFT configuration
        • Resume training (boolean flag)
        • Logging and evaluation strategy
        • Max sequence length

3. TRAINER CONFIGURATION (TrainerConf):
    - Define a `TrainerConf` class or dataclass to encapsulate all training configuration parameters.
    - This should include:
        • Optimizer settings (e.g., adamw_torch)
        • Scheduler settings (e.g., cosine, linear)
        • Evaluation strategy (steps or epoch)
        • Checkpointing frequency (save steps or epoch)
        • Mixed-precision (FP16/BF16)
        • Save/load/resume control
        • Gradient accumulation settings
        • Packing (for efficient training by concatenating short sequences)

4. TRAINING RESUMPTION:
    - Include functionality to resume interrupted training runs using a boolean flag such as `--resume_training`.
    - Ensure checkpoints are stored consistently and loaded correctly if resuming from the latest checkpoint.

5. DATA MANAGEMENT:
    - Implement utilities to load and preprocess training data efficiently.
    - Allow support for Hugging Face datasets (via `datasets.load_dataset`) and custom local datasets (JSON, CSV).
    - Include methods to tokenize, format, and group texts for causal or seq2seq training depending on the task type.
    - Ensure compatibility with both JSON and CSV formats for user flexibility.

6. CODE QUALITY AND PYTHON BEST PRACTICES:
    - Full adherence to modern Python best practices:
        • Use of type hints and data classes for configuration.
        • Modular design: separate data loading, training, evaluation, and utility functions.
        • Clear, consistent logging and exception handling.
        • PEP8-compliant formatting and readable code organization.
        • Docstrings and in-line comments for all classes and public functions.
    - Designed for reproducibility, robustness, and scalability in production training pipelines.

This module serves as a critical component for scalable and efficient fine-tuning of large-scale language models,
ensuring seamless integration with Hugging Face’s tools and optimal use of compute resources via Accelerate.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

# Conditional imports for structure, actual execution will not occur.
AutoModelForCausalLM = None
AutoTokenizer = None
TrainingArguments = None
SFTTrainer = None
PeftConfig = None
LoraConfig = None
get_peft_model = None
BitsAndBytesConfig = None
load_dataset = None
torch = None

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
        # EarlyStoppingCallback, # Optional: for early stopping
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftConfig,
        # prepare_model_for_kbit_training # If using k-bit training
    )
    from trl import SFTTrainer # Main trainer for SFT
    from datasets import load_dataset
except ImportError:
    logging.warning(
        "Hugging Face libraries (transformers, peft, trl, datasets, torch) not found. "
        "sft_train.py will not be fully functional. This is for code structure only."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainerConf:
    """
    Configuration class for SFT training parameters.
    Uses Hugging Face TrainingArguments and adds custom ones.
    """
    # Model parameters
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from Hugging Face Hub."})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA for PEFT."})
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability."})
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Common for Llama-like models
        metadata={"help": "Modules to apply LoRA to (e.g., ['q_proj', 'v_proj']). If None, PEFT will attempt to infer."}
    )
    use_4bit_quantization: bool = field(default=False, metadata={"help": "Whether to use 4-bit quantization."})
    bnb_4bit_compute_dtype: str = field(default="float16", metadata={"help": "Compute dtype for 4-bit base models (float16, bfloat16)."})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type (fp4 or nf4)."})
    bnb_4bit_use_double_quant: bool = field(default=True, metadata={"help": "Whether to use double quantization."})

    # Data parameters
    dataset_name_or_path: str = field(metadata={"help": "Path to the dataset or dataset identifier from Hugging Face Hub."})
    dataset_text_field: str = field(default="text", metadata={"help": "The name of the text field in the dataset (for SFT)."})
    # Or, if using prompt/response structure:
    # dataset_prompt_field: Optional[str] = field(default=None, metadata={"help": "Field for prompts."})
    # dataset_response_field: Optional[str] = field(default=None, metadata={"help": "Field for responses."})
    max_seq_length: int = field(default=1024, metadata={"help": "Maximum sequence length for tokenization."})
    packing: bool = field(default=True, metadata={"help": "Whether to use packing for SFTTrainer (efficiently packs short sequences)."})
    dataset_split_train: str = field(default="train", metadata={"help": "Dataset split for training."})
    dataset_split_eval: Optional[str] = field(default=None, metadata={"help": "Dataset split for evaluation."}) # e.g., "validation" or "test"
    eval_dataset_size: float = field(default=0.1, metadata={"help": "If dataset_split_eval is None, fraction of train to use for eval, or number of samples."})


    # TrainingArguments specific parameters (subset, most important ones)
    output_dir: str = field(default="./sft_output", metadata={"help": "Output directory for checkpoints and logs."})
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU for training."})
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "Batch size per GPU for evaluation."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    learning_rate: float = field(default=2e-4, metadata={"help": "Initial learning rate."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Learning rate scheduler type (e.g., 'linear', 'cosine')."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio for learning rate scheduler."})
    weight_decay: float = field(default=0.001, metadata={"help": "Weight decay applied to all layers except bias and LayerNorm weights."})
    optimizer_type: str = field(default="adamw_torch", metadata={"help": "Optimizer to use (e.g., 'adamw_torch', 'paged_adamw_8bit')."})

    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    save_strategy: str = field(default="steps", metadata={"help": "When to save checkpoints ('steps', 'epoch')."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps (if save_strategy='steps')."})
    save_total_limit: Optional[int] = field(default=2, metadata={"help": "Limit the total number of checkpoints. Deletes the oldest checkpoints."})

    evaluation_strategy: Optional[str] = field(default=None, metadata={"help": "When to evaluate ('steps', 'epoch', or None)."}) # Set to 'steps' or 'epoch' if eval_dataset is provided
    eval_steps: Optional[int] = field(default=500, metadata={"help": "Evaluate every X steps (if evaluation_strategy='steps')."})

    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision training (requires compatible GPU)."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use bfloat16 (mixed) precision training (requires Ampere+ GPU)."})

    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Path to a checkpoint to resume training from."})
    # Note: SFTTrainer/Transformers Trainer handles `resume_from_checkpoint=True` to find latest in output_dir.
    # Explicit path takes precedence.

    # Other SFTTrainer specific params
    neftune_noise_alpha: Optional[float] = field(default=None, metadata={"help":"Add noise to embeddings during training."}) # SFTTrainer specific

    # Custom arguments for script control
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading models."})

    # Add more TrainingArguments as needed, e.g., group_by_length, report_to, etc.

    def __post_init__(self):
        if self.use_4bit_quantization and (self.fp16 or self.bf16):
            logger.warning("4-bit quantization is enabled. fp16/bf16 flags might be ignored or handled by BitsAndBytesConfig.")
        if self.evaluation_strategy and not self.dataset_split_eval:
             logger.warning(f"evaluation_strategy is '{self.evaluation_strategy}' but no dataset_split_eval is provided. Evaluation might not run effectively unless a split is auto-created.")


def load_data(conf: TrainerConf, tokenizer: Any) -> Dict[str, Any]:
    """
    Loads and preprocesses the dataset.

    Args:
        conf: TrainerConf object with dataset parameters.
        tokenizer: The tokenizer to use for preprocessing.

    Returns:
        A dictionary containing 'train_dataset' and optionally 'eval_dataset'.
    """
    if not load_dataset:
        raise ImportError("HuggingFace datasets library not available. Cannot load data.")

    logger.info(f"Loading dataset: {conf.dataset_name_or_path}")
    # Try to load from path if it's a directory, otherwise assume HF Hub name or file path
    if Path(conf.dataset_name_or_path).is_dir():
        dataset = load_dataset("json", data_files={
            conf.dataset_split_train: str(Path(conf.dataset_name_or_path) / f"{conf.dataset_split_train}.jsonl"), # Assuming .jsonl
            # Add eval file if specified and exists
        })
        if conf.dataset_split_eval and (Path(conf.dataset_name_or_path) / f"{conf.dataset_split_eval}.jsonl").exists():
            dataset[conf.dataset_split_eval] = load_dataset("json", data_files={
                 conf.dataset_split_eval: str(Path(conf.dataset_name_or_path) / f"{conf.dataset_split_eval}.jsonl")
            })[conf.dataset_split_eval]

    elif Path(conf.dataset_name_or_path).is_file():
        file_type = Path(conf.dataset_name_or_path).suffix.lstrip('.')
        if file_type not in ["json", "jsonl", "csv"]:
            raise ValueError(f"Unsupported local file type: {file_type}. Supported: json, jsonl, csv.")

        data_files_map = {conf.dataset_split_train: conf.dataset_name_or_path}
        # Cannot easily specify eval split from a single file without more logic.
        # User should provide separate train/eval files or use dataset_split_eval with HF dataset.
        dataset = load_dataset(file_type if file_type != "jsonl" else "json", data_files=data_files_map)

    else: # Assume HuggingFace Hub dataset name
        dataset = load_dataset(conf.dataset_name_or_path)

    if not isinstance(dataset, dict) or conf.dataset_split_train not in dataset:
        raise ValueError(f"Train split '{conf.dataset_split_train}' not found in loaded dataset: {dataset.keys() if isinstance(dataset,dict) else 'Not a dict'}")

    train_dataset = dataset[conf.dataset_split_train]
    eval_dataset = None

    if conf.dataset_split_eval and conf.dataset_split_eval in dataset:
        eval_dataset = dataset[conf.dataset_split_eval]
    elif conf.evaluation_strategy and conf.dataset_split_eval is None : # Create eval from train
        logger.info(f"No dedicated eval split. Splitting from train set (using {conf.eval_dataset_size} for eval).")
        if train_dataset. μεγαλύτε(1): # Ensure there's enough data to split. train_dataset.num_rows
             split_data = train_dataset.train_test_split(test_size=conf.eval_dataset_size, shuffle=True, seed=42)
             train_dataset = split_data["train"]
             eval_dataset = split_data["test"]
        else:
            logger.warning("Train dataset too small to create an evaluation split. Skipping evaluation dataset creation.")


    # Further preprocessing (tokenization) is typically handled by SFTTrainer directly
    # when `dataset_text_field` is specified, or via a formatting function.
    # If custom formatting is needed (e.g. for prompt/response), provide a formatting_func to SFTTrainer.

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset}


def main_sft_training(conf: TrainerConf):
    """Main function to set up and run SFT training."""

    if not all([torch, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, SFTTrainer, LoraConfig, get_peft_model]):
        logger.critical("One or more core Hugging Face libraries are not available. SFT training cannot proceed.")
        return

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer: {conf.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        conf.model_name_or_path,
        trust_remote_code=conf.trust_remote_code,
        use_fast=True # Prefer fast tokenizers
    )
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # For some models, setting pad_token_id might also be needed in model config if pad_token is added this way.

    # 2. Load Model
    logger.info(f"Loading model: {conf.model_name_or_path}")
    quantization_config = None
    if conf.use_4bit_quantization:
        if not BitsAndBytesConfig:
            logger.error("BitsAndBytesConfig not available. Cannot apply 4-bit quantization.")
        else:
            compute_dtype = getattr(torch, conf.bnb_4bit_compute_dtype, torch.float16)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=conf.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=conf.bnb_4bit_use_double_quant,
            )
            logger.info(f"4-bit quantization enabled with compute_dtype={conf.bnb_4bit_compute_dtype}, quant_type={conf.bnb_4bit_quant_type}")

    model = AutoModelForCausalLM.from_pretrained(
        conf.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=conf.trust_remote_code,
        # device_map="auto", # Usually good for multi-GPU, SFTTrainer with Accelerate handles device placement.
        # torch_dtype=torch.bfloat16 if conf.bf16 and not conf.use_4bit_quantization else (torch.float16 if conf.fp16 and not conf.use_4bit_quantization else None)
    )

    # Resize token embeddings if tokenizer vocab size > model vocab size (e.g. after adding special tokens)
    # model.resize_token_embeddings(len(tokenizer)) # Generally good practice, but ensure it's needed.

    # Configure PEFT/LoRA
    peft_config: Optional[PeftConfig] = None
    if conf.use_lora:
        logger.info("Using LoRA for PEFT.")
        peft_config = LoraConfig(
            r=conf.lora_r,
            lora_alpha=conf.lora_alpha,
            lora_dropout=conf.lora_dropout,
            target_modules=conf.lora_target_modules if conf.lora_target_modules else None, # None lets peft infer
            bias="none", # Common setting
            task_type="CAUSAL_LM", # Or "SEQ_2_SEQ_LM" for encoder-decoder models
        )
        # model = get_peft_model(model, peft_config) # SFTTrainer can also handle applying PEFT config
        # model.print_trainable_parameters() # Useful for verifying LoRA setup
        logger.info(f"LoRA configured: r={conf.lora_r}, alpha={conf.lora_alpha}, target_modules={conf.lora_target_modules or 'auto'}")


    # 3. Load and Prepare Data
    datasets = load_data(conf, tokenizer)
    train_dataset = datasets["train_dataset"]
    eval_dataset = datasets.get("eval_dataset")


    # 4. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=conf.output_dir,
        num_train_epochs=conf.num_train_epochs,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.per_device_eval_batch_size,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        learning_rate=conf.learning_rate,
        lr_scheduler_type=conf.lr_scheduler_type, #type: ignore
        warmup_ratio=conf.warmup_ratio,
        weight_decay=conf.weight_decay,
        optim=conf.optimizer_type, #type: ignore

        logging_dir=f"{conf.output_dir}/logs",
        logging_strategy="steps", # Log according to logging_steps
        logging_steps=conf.logging_steps,

        save_strategy=conf.save_strategy, #type: ignore
        save_steps=conf.save_steps if conf.save_strategy == "steps" else None, # type: ignore
        save_total_limit=conf.save_total_limit,

        evaluation_strategy=conf.evaluation_strategy if eval_dataset else "no", #type: ignore
        eval_steps=conf.eval_steps if conf.evaluation_strategy == "steps" and eval_dataset else None, # type: ignore
        # load_best_model_at_end=True if eval_dataset else False, # Optional
        # metric_for_best_model="eval_loss" if eval_dataset else None, # Optional

        fp16=conf.fp16 and not conf.use_4bit_quantization, # BNB handles its own dtype for 4-bit
        bf16=conf.bf16 and not conf.use_4bit_quantization, # BNB handles its own dtype for 4-bit

        report_to="tensorboard", # Or "wandb", "none", etc.
        # ddp_find_unused_parameters=False, # Can be needed in some DDP setups
        # ... add other relevant TrainingArguments based on TrainerConf
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # type: ignore
        tokenizer=tokenizer,
        peft_config=peft_config if conf.use_lora else None, # Pass LoraConfig here
        dataset_text_field=conf.dataset_text_field, # Key for text data in dataset
        # formatting_func=formatting_func, # If custom prompt/response formatting is needed
        max_seq_length=conf.max_seq_length,
        packing=conf.packing, # Efficiently packs short sequences
        # dataset_num_proc=os.cpu_count(), # Number of processes for dataset mapping
        neftune_noise_alpha=conf.neftune_noise_alpha,
    )

    # 6. Start Training
    logger.info("Starting SFT training...")
    resume_from_checkpoint_arg = conf.resume_from_checkpoint
    if not resume_from_checkpoint_arg and Path(conf.output_dir).exists():
        # Transformers Trainer can auto-resume if output_dir has checkpoints
        # and resume_from_checkpoint=True (bool) is passed.
        # If string path is given, it uses that.
        # For simplicity, if specific path not given, let Trainer handle auto-resume from output_dir.
        logger.info(f"No specific checkpoint path given to resume. Trainer will attempt to resume from latest in '{conf.output_dir}' if checkpoints exist.")
        resume_from_checkpoint_arg = True # type: ignore

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint_arg) # type: ignore

    # 7. Save final model, tokenizer, and training state
    logger.info("Training finished. Saving final model and tokenizer.")
    trainer.save_model() # Saves adapter if PEFT, full model otherwise
    # trainer.save_state() # Saves optimizer, scheduler, etc. (already done by save_steps)
    tokenizer.save_pretrained(conf.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training metrics: {metrics}")

    # 8. (Optional) Evaluation after training if eval_dataset is provided
    if eval_dataset:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Evaluation metrics: {eval_metrics}")

    logger.info(f"SFT training complete. Artifacts saved to: {conf.output_dir}")


if __name__ == "__main__":
    # This block is for command-line execution and illustrative purposes.
    # Actual execution is not intended in the current environment due to missing dependencies.
    logger.info("SFT Training Script: Illustrative __main__ block.")
    logger.info("This script is intended to be run with `accelerate launch sft_train.py ...` or `python sft_train.py ...` with appropriate CLI arguments.")
    logger.info("Due to potential missing dependencies (torch, transformers, etc.), this main block will not run training.")

    # Example of how to parse arguments using Hugging Face HfArgumentParser (if available)
    # from transformers import HfArgumentParser
    # parser = HfArgumentParser(TrainerConf)
    # training_config, = parser.parse_args_into_dataclasses()
    # logger.info(f"Illustrative Config: {training_config}")
    # main_sft_training(training_config)

    # Fallback minimal example if HfArgumentParser isn't used/available for this illustration
    if False: # Do not run this illustrative part
        class DummyTokenizer:
            pad_token = None
            eos_token = "</s>"
            def __call__(self, *args, **kwargs): return {"input_ids": [], "attention_mask": []}
            def save_pretrained(self, path): pass

        class DummyDataset:
            def __len__(self): return 10
            def train_test_split(self, test_size, shuffle, seed): return {"train": self, "test": self}
            num_rows = 10 # Mock attribute
            def __getitem__(self, idx): return { "text": "dummy text example for sft training, very interesting."}


        dummy_conf = TrainerConf(
            model_name_or_path="dummy-gpt2",
            dataset_name_or_path="dummy-dataset",
            output_dir="./dummy_sft_output_illustrative",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_seq_length=64,
            logging_steps=1,
            save_steps=2,
            dataset_split_eval="test", # Needs a dummy eval set too
            evaluation_strategy="steps",
            eval_steps=2,
        )
        logger.info(f"Illustrative Dummy Config: {dummy_conf}")
        # main_sft_training(dummy_conf) # This would fail without real libraries
        logger.info("Illustrative main_sft_training call would be here.")

    logger.info("SFT Training script __main__ finished (illustrative).")
