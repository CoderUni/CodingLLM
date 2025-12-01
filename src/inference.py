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

# Keep the imports here. Otherwise, the above env vars may not take effect.
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_DIR = "REPLACE_WITH_YOUR_FINETUNED_MODEL_PATH"

# ========================================
# Insert your prompt/coding question here
# ========================================
PROMPT = """
Insert prompt here
"""

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=config["model"]["max_seq_length"],
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=config["model"]["max_seq_length"],
            temperature=config["model"]["temperature"],
            top_p=config["model"]["top_p"],
            top_k=config["model"]["top_k"],
            streamer=TextStreamer(tokenizer, skip_prompt=False),
        )
