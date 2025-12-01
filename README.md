<h1 align="center">
<img src="assets/logo.png" alt="Anni Logo" width="100" />
<br />
Anni
</h1>

<p align="center">
<a href="https://huggingface.co/BigJuicyData/Anni" target="_blank"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Anni-ffc107?color=ffc107&logoColor=white"/></a>
<a href="https://modelscope.cn/models/quanteat/Anni" target="_blank">
    <img alt="ModelScope Model" src="https://img.shields.io/badge/🤖%20ModelScope-Anni-604ad3?color=604ad3"/>
</a>
<a href="https://github.com/CoderUni/CodingLLM/actions/workflows/codeql.yml">
  <img src="https://github.com/CoderUni/CodingLLM/actions/workflows/codeql.yml/badge.svg" alt="Build Status">
</a>
</p>

<p align="center">
<strong>Anni</strong> is a high-performance code assistant built upon the <strong>Qwen3 14B</strong> architecture. Fine-tuned on the <strong>OpenCodeReasoning-2</strong> dataset, Anni is engineered to excel in deep algorithmic reasoning, competitive programming logic, and the implementation of complex, high-efficiency data structures.
</p>

---

## 🚀 Model Overview

| Property | Value |
|---------|--------|
| Base Model | Qwen3 14B |
| Model Type | Language Model for Code |
| Context Length | 32,000 tokens |
| Precision | BF16 / safetensors (merged) |
| Inference Framework | vLLM compatible |

---

## 💻 Usage

**Get started immediately** using the provided Google Colab notebooks:

*   **(Recommended) GGUF Inference :** Open the [Colab Notebook](https://colab.research.google.com/drive/16RKUtphbI1rAds_lLwPGk2cRhf9CDJDo?usp=sharing)  to run standard inference.

*   **vLLM Serving:** Open the [Colab Notebook](https://colab.research.google.com/drive/1lXYtLT729qcxJPc56TllgwiGEsjIiW0Q?usp=sharing) to run inference using the vLLM server.

---

## 🛠️ Development Setup

### Prerequisites

1.  **Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **System Tools:**
    Ensure `tmux` is installed on your system (required for training scripts).

### Configuration

1.  **Environment Variables:**
    Rename the example environment file and add your API tokens (WandB, HuggingFace, ModelScope).
    ```bash
    mv config/example.env config/.env
    # Edit config/.env with your keys
    ```

2.  **Training Config:**
    Edit [config/config.yaml](config/config.yaml) to adjust hyperparameters.
    *   *Note:* Specify the `LOCAL_STORAGE_PATH` in [src/train.py](src/train.py) before starting training.

### Running Training

To start the training process, run the shell script:

```bash
./scripts/train.sh
```

---

## 📂 Project Structure

### Source (`src/`)
| File | Description |
|------|-------------|
| [`preprocess.py`](src/preprocess.py) | Downloads the [OpenCodeReasoning-2 dataset](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) and preprocesses it for training. |
| [`train.py`](src/train.py) | Downloads the base model and fine-tunes it on the preprocessed dataset. |
| [`save.py`](src/save.py) | Loads the fine-tuned LoRA adapters and saves the model as merged 16-bit and GGUF formats. |
| [`upload.py`](src/upload.py) | Uploads the merged model to Hugging Face and ModelScope. |

### Scripts (`scripts/`)
| File | Description |
|------|-------------|
| [`train.sh`](scripts/train.sh) | Runs the training script with specified parameters. |
| [`eval.sh`](scripts/eval.sh) | Evaluates the model on the LiveCodeBench dataset. |
| [`serve.sh`](scripts/serve.sh) | Serves the model using the vLLM server. |
| [`terminate_train.sh`](scripts/terminate_train.sh) | Terminates the training process. |

### Frontend (`web/`)
The frontend code for Anni is available in the `web` directory.
👉 **[View Frontend Documentation](web/README.md)**

---

## ⚖️ License

This repository’s **model and its training code** are released under the **MIT License**.  
All other elements, such as **frontend code, project name and logo**, are **trademarks** of the developer and owner of this repository (**Hans**) and **may not be used without explicit permission**.

---

## 📚 Training Dataset Notice

The training dataset includes openly licensed sources under **CC-BY-4.0**, which **permits commercial use with attribution**.

**Attribution:**

- [OpenCoderReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) (CC-BY-4.0)

> Note: The dataset itself is **not included** in this model release.
---

## ⚠️ Disclaimer

This model may generate incorrect or unsafe code.
Evaluate and verify outputs before using in production.
