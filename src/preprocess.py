import hashlib
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from tqdm import tqdm
import os
from pathlib import Path
import pickle
import random
from typing import Optional, Tuple


logger = get_logger()

# Load hf datasets later if needed
hf_datasets = {
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces"),
}

# From https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2
def get_question(ds_name, split, index):
    """
    Retrieves the question text from the specified dataset, split, and index.
    Args:
        ds_name (str): Name of the dataset (e.g., "code_contests", "taco", "apps", "open-r1/codeforces").
        split (str): Dataset
        index (int): Index of the benchmark within the split.
    Returns:
        str or None: The question text if found, otherwise None.
    """

    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


def analyze_token_lengths(
    tokenizer, dataset, dataset_cache_dir: str, text_column: str = "text"
):
    """
    Analyzes and plots the token length distribution of the given dataset.
    Saves a histogram plot to the dataset cache directory.
    """

    logger.info("Starting token length analysis on training data...")

    # We re-tokenize the final text string to get an accurate length count.
    def get_token_length(example):
        tokens = tokenizer(example["text"], truncation=False, padding=False)
        return {"token_length": len(tokens["input_ids"])}

    # Map dataset to get length
    logger.info(
        "Mapping dataset to get token lengths... (This might take a few minutes)"
    )
    dataset_with_lengths = dataset.map(
        get_token_length, num_proc=os.cpu_count() or 4 # Use available CPUs
    )

    # Analyze using Pandas
    logger.info("Analyzing token lengths...")
    df = dataset_with_lengths.to_pandas()

    # Get stats
    logger.info("Stats")
    logger.info(df["token_length"].describe())

    # Get Percentiles
    logger.info("Percentiles (Quantiles)")
    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    logger.info(df["token_length"].quantile(quantiles))

    # Plot histogram
    logger.info("\nGenerating histogram plot...")
    plt.figure(figsize=(12, 6))
    plt.hist(df["token_length"], bins=100, log=True)
    plt.title("Histogram of Token Lengths (Log Scale)")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True)

    # Save
    plot_path = os.path.join(
        Path(dataset_cache_dir).parent, "token_length_histogram.png"
    )
    plt.savefig(plot_path)
    logger.info(f"Saved histogram to {plot_path}")


def load_and_tokenize(
    base_tokenizer,
    dataset_cache_dir: str,
    test_ratio: float = 0.02,
    check_token_length: bool = False,
    min_seq_length: int | None = 0,
    max_seq_length: int | None = 0,
    previous_ids_path: str | None = None,
    apply_token_balancing: bool = False,
    limit_samples: Optional[int] = None,
) -> Tuple:
    """
    Loads, tokenizes, filters by length/used IDs, and optionally balances the dataset.
    """
    logger.info(
        f"--- Load & Tokenize: Min={min_seq_length}, Max={max_seq_length}, Limit={limit_samples}"
    )

    tokenizer = get_chat_template(base_tokenizer, chat_template="qwen3-thinking")

    # Load or download dataset
    cache_path = Path(dataset_cache_dir)
    dataset_path = os.path.join(cache_path.parent, "OpenCodeReasoning-2-filled")

    if os.path.exists(dataset_path):
        logger.info("Found filled dataset cache, loading from disk...")
        dataset = load_from_disk(os.path.join(dataset_path))
        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
    else:
        logger.info(
            "No filled dataset cache found, downloading and processing dataset..."
        )
        check_token_length = True  # Ensure analysis runs after download
        train_dataset, valid_dataset = download_dataset(
            dataset_cache_dir=dataset_cache_dir, test_ratio=test_ratio
        )

    # Generate conversation pairs & Apply chat template
    train_dataset = train_dataset.map(
        generate_conversation, batched=True, desc="Formatting train conversations"
    )
    valid_dataset = valid_dataset.map(
        generate_conversation, batched=True, desc="Formatting valid conversations"
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(
        formatting_prompts_func, batched=True, desc="Applying chat template to train"
    )
    valid_dataset = valid_dataset.map(
        formatting_prompts_func, batched=True, desc="Applying chat template to valid"
    )

    # Calculate lengths & Generate IDs (Modified for Unique ID)
    def process_metadata(example):
        text = example.get("text", "")
        return {
            "token_length": len(tokenizer.encode(text)),
            "unique_id": hashlib.md5(text.encode("utf-8")).hexdigest(),
        }

    train_dataset = train_dataset.map(process_metadata, num_proc=os.cpu_count())
    valid_dataset = valid_dataset.map(process_metadata, num_proc=os.cpu_count())

    # (ONLY FOR STAGE 4) Exclude previously used IDs
    used_ids_set = set()
    if previous_ids_path and os.path.exists(previous_ids_path):
        with open(previous_ids_path, "rb") as f:
            used_ids_set = pickle.load(f)
        logger.info(f"Loaded {len(used_ids_set)} used IDs for exclusion.")

        # Filter train dataset
        initial_train_len = len(train_dataset)
        train_dataset = train_dataset.filter(
            lambda x: x["unique_id"] not in used_ids_set,
            num_proc=os.cpu_count(),
            desc="Removing previously used data from train",
        )
        logger.info(f"Train: Removed {initial_train_len - len(train_dataset)} samples.")

        # Filter valid dataset (Crucial for Stage 4 fresh validation data)
        initial_valid_len = len(valid_dataset)
        valid_dataset = valid_dataset.filter(
            lambda x: x["unique_id"] not in used_ids_set,
            num_proc=os.cpu_count(),
            desc="Removing previously used data from valid",
        )
        logger.info(f"Valid: Removed {initial_valid_len - len(valid_dataset)} samples.")

    # Filter by length (Using filter helper directly)
    logger.info(f"Applying length filter: min={min_seq_length}, max={max_seq_length}")

    def filter_by_length(example):
        length = example["token_length"]
        if min_seq_length is not None and length < min_seq_length:
            return False
        if max_seq_length is not None and length > max_seq_length:
            return False
        return True

    train_dataset = train_dataset.filter(filter_by_length, num_proc=os.cpu_count())
    valid_dataset = valid_dataset.filter(filter_by_length, num_proc=os.cpu_count())

    # Limit samples if specified (for simulation stages)
    if limit_samples is not None and len(train_dataset) > limit_samples:
        train_dataset = train_dataset.select(
            range(len(train_dataset) - limit_samples, len(train_dataset))
        )
        logger.info(
            f"Limiting train dataset to the last {limit_samples} samples for ID tracking."
        )

    # Apply token balancing (Only for stage 4)
    if apply_token_balancing:
        train_dataset = balance_by_token_share(train_dataset, logger)

    # We save IDs only if we are simulating (limit_samples is set).
    if previous_ids_path and limit_samples is not None:
        # Collect IDs from both the limited train set and the full valid set of this stage
        current_ids = set(train_dataset["unique_id"])
        current_ids.update(set(valid_dataset["unique_id"]))

        combined_ids = used_ids_set.union(current_ids)

        with open(previous_ids_path, "wb") as f:
            pickle.dump(combined_ids, f)
        logger.info(
            f"Updated {previous_ids_path}. Total used IDs tracked: {len(combined_ids)}"
        )

    logger.info(
        f"Final Train Size: {len(train_dataset)}, Valid Size: {len(valid_dataset)}"
    )

    logger.info("Sample formatted prompt (First 500 chars):")
    if len(train_dataset) > 0:
        logger.info(train_dataset[0]["text"][:500])
    else:
        logger.info("Training dataset is empty after filtering!")

    if check_token_length:
        analyze_token_lengths(
            tokenizer, train_dataset, dataset_cache_dir, text_column="text"
        )

    return tokenizer, train_dataset, valid_dataset


def download_dataset(dataset_cache_dir: str, test_ratio: float = 0.02):
    """
    Downloads the OpenCodeReasoning-2 dataset, fills missing questions,
    and splits into train/validation sets.
    """

    logger.info("Downloading OpenCodeReasoning-2 dataset...")

    # Load dataset
    dataset = load_dataset(
        "nvidia/OpenCodeReasoning-2",
        cache_dir=dataset_cache_dir,
    )

    # Print dataset
    logger.info("\n--- Dataset Info")
    logger.info(dataset)

    def populate_question(example):
        if example["dataset"] in ["code_contests", "open-r1/codeforces"]:
            ds_name, ds_split, ds_index = (
                example["dataset"],
                example["split"],
                int(example["index"]),
            )

            question = get_question(ds_name, ds_split, ds_index)

            # Populate the question field if get_question was successful
            if question is not None and question.strip() != "":
                example["question"] = question
        return example

    # Filter dataset to only those with pass_rate == 1.0 and not from taco/apps
    logger.info("Filtering dataset...")
    filtered_dataset = dataset["python"].filter(
        lambda x: (
            x["pass_rate"] == "1.0"  # Use the correct string "1.0"
            and x["dataset"] not in ["taco", "apps"]
        ),
        num_proc=os.cpu_count(),  # Use multiple cores to speed up
    )
    logger.info(f"Filtered dataset size: {len(filtered_dataset)}")

    # This applies the 'populate_question' function to every row
    # in 'filtered_dataset' and saves the result.
    logger.info("Populating questions...")
    populated_dataset = filtered_dataset.map(
        populate_question,
        desc="Populating questions",
        num_proc=os.cpu_count(),  # Use multiple cores to speed up
    )

    # Train/validation split
    # Use the 'populated_dataset' which now has the correct questions
    split_data = populated_dataset.train_test_split(test_size=test_ratio, seed=42)
    train_dataset = split_data["train"]
    valid_dataset = split_data["test"]

    # Wrap splits into a DatasetDict (for easy saving)
    dataset_dict = DatasetDict({"train": train_dataset, "validation": valid_dataset})

    # Save to disk (using Apache Arrow)
    cache_path = Path(dataset_cache_dir)
    save_path = os.path.join(cache_path.parent, "OpenCodeReasoning-2-filled")
    dataset_dict.save_to_disk(save_path)
    logger.info(f"Saved dataset with filled questions to: {save_path}")

    return train_dataset, valid_dataset


# See https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
def generate_conversation(examples):
    """
    Generates conversation pairs from question and solution fields.
    Assumes examples is a batch of data.
    """

    problems = examples["question"]
    solutions = examples["r1_generation"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        if not problem or problem.strip() == "":
            continue  # skip empty questions
        conversations.append(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        )
    return {"conversations": conversations}


def sum_tokens(ds):
    """Calculates the total number of tokens in a dataset split."""
    return sum(ds["token_length"])


def balance_by_token_share(dataset, logger):
    """
    Filters dataset to match target token distribution:
    Short (<=4k): 35%, Mid (4k-14k): 35%, Long (>14k): 30%
    """
    logger.info("Balancing dataset for Stage 4 Token Share targets...")

    # Token length ranges and target ratios
    ranges = {
        "Short": (0, 4096, 0.35),
        "Mid": (4097, 13946, 0.35),
        "Long": (13947, 32000, 0.30),
    }

    datasets = {}
    tokens_available = {}

    # Split into buckets and calculate available tokens
    for name, (min_len, max_len, _) in ranges.items():
        ds = dataset.filter(lambda x: min_len <= x["token_length"] <= max_len)
        datasets[name] = ds
        tokens_available[name] = sum_tokens(ds)

    logger.info(
        f"Available Tokens - Short: {tokens_available['Short']:,}, Mid: {tokens_available['Mid']:,}, Long: {tokens_available['Long']:,}"
    )

    if all(t == 0 for t in tokens_available.values()):
        logger.info("No tokens available in any bucket. Returning empty dataset.")
        return dataset.select([])

    # Calculate the total budget based on the limiting factor
    max_budgets = {}
    for name, (_, _, ratio) in ranges.items():
        max_budgets[name] = tokens_available[name] / ratio if ratio > 0 else 0

    positive_budgets = [val for val in max_budgets.values() if val > 0]
    if not positive_budgets:
        logger.info("Ratios cannot be met.")
        return dataset.shuffle(seed=42)

    total_budget = min(positive_budgets)

    # Calculate target tokens based on the minimum budget
    target_tokens = {}
    for name, (_, _, ratio) in ranges.items():
        target_tokens[name] = int(total_budget * ratio)

    logger.info(
        f"Target Tokens - Short: {target_tokens['Short']:,}, Mid: {target_tokens['Mid']:,}, Long: {target_tokens['Long']:,}"
    )

    # Helper to select samples until token count is met
    def select_samples(ds, target_tokens):
        current_tokens = 0
        selected_indices = []
        indices = list(range(len(ds)))
        random.seed(42)
        random.shuffle(indices)

        for idx in indices:
            length = ds[idx]["token_length"]
            # Stop if the next addition exceeds the target significantly
            if (
                current_tokens + length > target_tokens
                and current_tokens > target_tokens * 0.95
            ):
                continue

            current_tokens += length
            selected_indices.append(idx)

            if current_tokens >= target_tokens:
                break

        return ds.select(selected_indices)

    # Select and Concatenate
    from datasets import concatenate_datasets

    final_datasets = [
        select_samples(datasets[name], target_tokens[name]) for name in ranges.keys()
    ]
    balanced_dataset = concatenate_datasets(final_datasets)

    logger.info(f"Final Tokens: {sum_tokens(balanced_dataset):,}. Target achieved.")

    return balanced_dataset.shuffle(seed=42)

def simulate_used_data(base_tokenizer, dataset_cache_dir, proxies, used_ids_file):
    """
    Simulates the data consumption of Stages 1, 2, and 3 to collect all used IDs
    and save them to used_ids_file.

    **SINCE I *didn't* :( save the used IDs during the initial runs of Stages 1-3,
    we need to re-run the data loading with ID tracking enabled to reconstruct
    the used IDs file for Stage 4 exclusion.**
    """
    logger.info("=============================================")
    logger.info("  STAGE 1 SIMULATION (SHORT: 0-4096 tokens)  ")
    logger.info("=============================================")

    stage1_proxies, stage1_test_ratio = proxies["stage_1"]
    load_and_tokenize(
        base_tokenizer,
        dataset_cache_dir=dataset_cache_dir,
        min_seq_length=0,
        max_seq_length=4096,
        test_ratio=stage1_test_ratio,
        previous_ids_path=used_ids_file,
        apply_token_balancing=False,  # Must be False during simulation
        limit_samples=stage1_proxies,
    )

    logger.info("=================================================")
    logger.info("  STAGE 2 SIMULATION (SHORT: 4097-13946 tokens)  ")
    logger.info("=================================================")

    stage2_proxies, stage2_test_ratio = proxies["stage_2"]
    load_and_tokenize(
        base_tokenizer,
        dataset_cache_dir=dataset_cache_dir,
        min_seq_length=4097,
        max_seq_length=13946,
        test_ratio=stage2_test_ratio,
        previous_ids_path=used_ids_file,
        apply_token_balancing=False,  # Must be False during simulation
        limit_samples=stage2_proxies,
    )

    logger.info("=================================================")
    logger.info("  STAGE 3 SIMULATION (LONG: 13947-32000 tokens)  ")
    logger.info("=================================================")

    stage3_proxies, stage3_test_ratio = proxies["stage_3"]
    load_and_tokenize(
        base_tokenizer,
        dataset_cache_dir=dataset_cache_dir,
        min_seq_length=13947,
        max_seq_length=32000,
        test_ratio=stage3_test_ratio,
        previous_ids_path=used_ids_file,
        apply_token_balancing=False,  # Must be False during simulation
        limit_samples=stage3_proxies,
    )

    logger.info(
        "Done simulating the used datasets. Used IDs are tracked in cumulative_used_data_ids.pkl"
    )


def get_stage4_datasets(
    base_tokenizer,
    min_seq_length,
    max_seq_length,
    test_ratio,
    dataset_cache_dir,
    proxies,
    used_ids_file,
):
    """
    Orchestrates the simulation and then loads the final Stage 4 datasets
    with exclusion and token balancing.

    Returns:
        tokenizer, train_dataset, valid_dataset
    """
    # Run the simulation to create the used_ids_file containing all used IDs
    simulate_used_data(base_tokenizer, dataset_cache_dir, proxies, used_ids_file)

    # 2. Run the final Stage 4 load
    logger.info("======================================")
    logger.info("  STAGE 4: FINAL DATASET PREPARATION  ")
    logger.info("======================================")

    # Load all data (0-32k), filter out all used IDs, and apply token share balancing.
    tokenizer, train_dataset, valid_dataset = load_and_tokenize(
        base_tokenizer,
        dataset_cache_dir=dataset_cache_dir,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        test_ratio=test_ratio,
        previous_ids_path=used_ids_file,  # EXCLUDE all previously used data
        apply_token_balancing=True,  # Apply 35/35/30 split
    )

    # The original validation set is most likely close to empty because it was already used in Stages 1-3.
    # so we create a new validation split from the fresh training data.
    target_valid_size = int(len(train_dataset) * test_ratio)

    if len(valid_dataset) < target_valid_size:
        logger.info(
            f"Existing valid set is too small ({len(valid_dataset)} < {target_valid_size})."
        )
        logger.info(
            f"Merging surviving valid data back into train and creating a new {test_ratio*100}% split..."
        )

        # Merge surviving valid data back into train so we don't lose it
        if len(valid_dataset) > 0:
            train_dataset = concatenate_datasets([train_dataset, valid_dataset])

        split_data = train_dataset.train_test_split(test_size=test_ratio, seed=42)
        train_dataset = split_data["train"]
        valid_dataset = split_data["test"]

        logger.info(
            f"New Split Sizes -> Train: {len(train_dataset)}, Valid: {len(valid_dataset)}"
        )
    else:
        logger.info(f"Using surviving validation data. Size: {len(valid_dataset)}")

    # Clean up the temporary ID file
    try:
        os.remove(used_ids_file)
        logger.info(f"Cleaned up temporary ID file: {used_ids_file}")
    except OSError as e:
        logger.info(f"Could not remove temporary ID file: {e}")

    return tokenizer, train_dataset, valid_dataset
