# Bring imports
import os
from pathlib import Path
from src.utils.config_loader import config

# Specify local storage path
LOCAL_STORAGE_PATH = "/mnt/storage/metnet/coding_llm"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

# WanDB configuration
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = config["logging"]["wandb_project"]
os.environ["WANDB_LOG_MODEL"] = config["logging"]["wandb_log_model"]
WANDB_NAME=config["logging"]["run_name"]

# Set cache directory for HuggingFace and Modelscope
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")

# =============================================
# If you are from China, uncomment these lines.
# =============================================

# os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
# os.environ["MODELSCOPE_CACHE"] = os.path.join(CACHE_DIR, "modelscope")
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Use HuggingFace mirror for China
# os.environ["HF_HUB_OFFLINE"] = "0"  # Allow online access through mirror

from src.utils.logger import setup_logger

logger = setup_logger(log_dir="../logs")

import shutil
from unsloth import FastLanguageModel, unsloth_train
import torch
import wandb
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer, EarlyStoppingCallback

from src.preprocess import load_and_tokenize, get_stage4_datasets

# Set this to saved LORA adapter path if you want to resume training
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

MIN_SEQ_LENGTH = 0  # Stage 1 : 0, Stage 2: 4097, Stage 3: 13947
MAX_SEQ_LENGTH = 32000  # Stage 1: 4096, Stage 2: 13946, Stage 3 and 4: 32000

# Set the following to True if resuming from checkpoint.
# See https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint
RESUME_FROM_CHECKPOINT = False 

# Output directory to save model checkpoints and final model
OUTPUT_DIR = os.path.join(LOCAL_STORAGE_PATH, "stage4_final_checkpoints")

def clear_cache():
    """
    Clears the cache folder `.cache` to free up space.
    Use with caution as it will delete all cached files.
    This only deletes the *huggingface* and *modelscope* model cache folders.
    To clear the dataset cache, delete the dataset cache folder `data` separately.
    """
    local_cache = LOCAL_STORAGE_PATH
    if os.path.exists(local_cache):
        logger.info(f"Clearing local cache: {local_cache}")
        shutil.rmtree(local_cache)
        logger.info("Cache cleared!")


def download_model():
    """
    Downloads the pre-trained model and tokenizer from the specified MODEL_NAME.
    Returns:
        - `model`: The downloaded FastLanguageModel.
        - `tokenizer`: The corresponding tokenizer.
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,  # if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True, # QLora / 4-bit quantization
        load_in_8bit=False,
        full_finetuning=False,
        device_map=None,
    )

    logger.info(f"Model and tokenizer loaded successfully.")

    return model, tokenizer

def set_model_parameters(baseModel):
    logger.info("Setting model parameters for LoRA fine-tuning...")
    return FastLanguageModel.get_peft_model(
        baseModel,
        r=config["model"]["lora"]["r"],  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config["model"]["lora"]["alpha"],  # Best to choose alpha = rank or rank*2
        lora_dropout=config["model"]["lora"]["dropout"],  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=False,  # True or "unsloth" for very long context
        random_state=config["seed"],
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,
    )


def train_model(model, tokenizer, train_dataset, valid_dataset):
    # Genarete the docstring
    """
    Fine-tunes the given model using the provided training and validation datasets.
    Args:
        model: The FastLanguageModel to be fine-tuned.
        tokenizer: The corresponding tokenizer.
        train_dataset: The training dataset.
        valid_dataset: The validation dataset.
    Returns:
        - `model`: The fine-tuned FastLanguageModel.
        - `tokenizer`: The corresponding tokenizer.
    """

    logger.info("Starting training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        packing=True,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config["training"]["batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            learning_rate=config["training"]["learning_rate"],  # 2e-5 for stage 1-3, 1e-5 for stage 4
            weight_decay=config["training"]["weight_decay"],  # 0.01 for stage 1-3, 0.001 for stage 4
            optim="adamw_8bit",
            lr_scheduler_type="linear",  # cosine for stage 1-3, linear for stage 4
            # Short training
            warmup_ratio=config["training"]["warmup_ratio"],
            max_steps=config["training"]["max_steps"],
            # warmup_steps = 10,
            # num_train_epochs=1,  # Only for full training run
            # Eval
            eval_strategy="steps",
            eval_steps=config["eval"]["eval_steps"],
            per_device_eval_batch_size=config["eval"]["batch_size"],
            eval_accumulation_steps=config["eval"]["gradient_accumulation"],
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Saving
            save_strategy="steps",
            save_steps=config["save"]["steps"],
            save_total_limit=config["save"]["limit"],
            # Logging and output
            run_name=WANDB_NAME,  # Unique W&B run name
            logging_steps=config["logging"]["logging_steps"],
            output_dir=config["save"]["output_dir"],
            seed=config["seed"],
            report_to="wandb",
        ),
    )

    # Set early stopping parameters
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping"]["patience"],
        early_stopping_threshold=config["early_stopping"]["threshold"],
    )

    trainer.add_callback(early_stopping_callback)

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Verify masking part works:
    logger.info("Verifying that training data masking works...")
    logger.info(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))

    logger.info("Extra output here:")
    logger.info(
        tokenizer.decode(
            [
                tokenizer.pad_token_id if x == -100 else x
                for x in trainer.train_dataset[100]["labels"]
            ]
        ).replace(tokenizer.pad_token, " ")
    )

    # Show current memory stats
    logger.info("Showing initial GPU memory stats...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved.")

    # Start training
    logger.info("Starting training...")

    # Setup wandb logging
    run = wandb.init(project=os.getenv("WANDB_PROJECT"), name=WANDB_NAME)

    # Train
    trainer_stats = unsloth_train(
        trainer, resume_from_checkpoint=RESUME_FROM_CHECKPOINT
    )

    run.finish()
    logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']}s.")

    # Show final memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.info(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    return model, tokenizer


def save_model(model, tokenizer):
    """
    Saves the adapters of the fine-tuned model and tokenizer.
    Args:
        model: The fine-tuned FastLanguageModel.
        tokenizer: The corresponding tokenizer.
    """

    logger.info("Saving final model and tokenizer...")

    final_output_dir = os.path.join(LOCAL_STORAGE_PATH, "final_finetuned_model_lora")

    # Save adapters
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info(f"LORA adapters saved to: {final_output_dir}")

    logger.info(f"To save as GGUF or merged 16-bit model for VLLM, run src/save.py.")
    logger.info(f"Set FINETUNED_DIR = {final_output_dir} in src/save.py.")
    logger.info("Finished training model!")

if __name__ == "__main__":
    # Don't clear the cache unless you need to.
    # clear_cache()

    model, tokenizer = download_model()

    # I trained the model in 4 stages with different datasets.
    # In the final stage, I removed all samples used in the earlier stages.
    # For simplicity, simply run the stage 1-3 code.

    # For stage 1 to stage 3
    tokenizer, train_dataset, valid_dataset = load_and_tokenize(
       tokenizer,
       min_seq_length=MIN_SEQ_LENGTH,
       max_seq_length=MAX_SEQ_LENGTH,
       test_ratio=config["training"]["test_ratio"],
       dataset_cache_dir=os.path.join(LOCAL_STORAGE_PATH, "data/OpenCodeReasoning-2")
    )

    # For stage 4

    # Since we didn't track ids in earlier stages, we will simulate the data usage
    # Proxies for the number of raw samples consumed based on steps and batch sizes.
    # Approximated sample numbers are the worst case estimates.
    # The goal is to remove previously used samples (in the first 3 stages) from the final Stage 4 dataset.

    # proxies = {
    #     # 800 steps * 16 EBS * ~6 samples/packed-seq (Short)
    #     "stage_1": (76800, 0.2),  # (Samples, test ratio)
    #     # 600 steps * 16 EBS * ~4 samples/packed-seq (Mid)
    #     "stage_2": (38400, 0.2),
    #     # 200 steps * 16 EBS * ~3 samples/packed-seq (Long)
    #     "stage_3": (9600, 0.1),
    # }

    # File to store the cumulative unique IDs used across all stages
    # used_ids_file = "previously_used_data_ids.pkl"
 
    # tokenizer, train_dataset, valid_dataset = get_stage4_datasets(
    #     tokenizer,
    #     min_seq_length=MIN_SEQ_LENGTH,
    #     max_seq_length=MAX_SEQ_LENGTH,
    #     test_ratio=0.05,
    #     dataset_cache_dir=os.path.join(LOCAL_STORAGE_PATH, "data/OpenCodeReasoning-2"),
    #     proxies=proxies,
    #     used_ids_file=used_ids_file,
    # )

    model = set_model_parameters(model)

    model, tokenizer = train_model(model, tokenizer, train_dataset, valid_dataset)
    save_model(model, tokenizer)
