import os
from src.utils.config_loader import config

CACHE_DIR = "../.cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")

# If you are from China, uncomment these lines.
# os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
# os.environ["MODELSCOPE_CACHE"] = os.path.join(CACHE_DIR, "modelscope")
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Use HuggingFace mirror for China
# os.environ["HF_HUB_OFFLINE"] = "0"  # Allow online access through mirror

from modelscope.hub.api import HubApi


MODELSCOPE_ACCESS_TOKEN = os.getenv("MODELSCOPE_ACCESS_TOKEN")
api = HubApi()
api.login(MODELSCOPE_ACCESS_TOKEN)

HF_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")


import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Your finetuned model directory
# REPLACE THIS!
FINETUNED_DIR = ""

MAX_SEQ_LENGTH = 32000

if __name__ == "__main__":

    baseModel, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map=None,
    )

    # We are using the same parameters as the train script here
    model = FastLanguageModel.get_peft_model(
        baseModel,
        r=config["model"]["lora"]["r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config["model"]["lora"]["alpha"],
        lora_dropout=config["model"]["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing=False,
        random_state=config["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    # Save as merged 16-bit model for VLLM
    model.save_pretrained_merged(
        "final_finetuned_model", tokenizer, save_method="merged_16bit"
    )

    # See https://docs.unsloth.ai/basics/inference-and-deployment to check the other saving options

    # ===============================================
    # SET TO True to enable the saving options below
    # ===============================================

    # Save to q4_k_m GGUF
    if False:
        model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

    # Push to HuggingFace Hub as GGUF
    # Remember to set your huggingface access token in the .env file!
    if False:
        model.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="q4_k_m", token=HF_ACCESS_TOKEN
        )
