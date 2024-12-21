import logging
import os
import traceback
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional

# Extra imports for comparisons
try:
    import tiktoken  # For OpenAI's GPT tokenization
except ImportError:
    tiktoken = None
    logging.warning("tiktoken not installed. Run: pip install tiktoken")

try:
    import sentencepiece as spm  # For Google's SentencePiece
except ImportError:
    spm = None
    logging.warning("sentencepiece not installed. Run: pip install sentencepiece")

from tokenizers import Tokenizer  # This loads your MedToken from a JSON file

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tokens/validation.log"),
        logging.StreamHandler()
    ]
)

###############################################################################
# Plotting Functions (Existing)
###############################################################################
def plot_sequence_lengths(original_lengths: List[int], truncated_lengths: List[int]):
    """Plots original vs. truncated sequence lengths."""
    plt.figure(figsize=(10, 6))
    plt.plot(original_lengths, label="Original Lengths", marker="o")
    plt.plot(truncated_lengths, label="Truncated Lengths", marker="x")
    plt.xlabel("Sample Index")
    plt.ylabel("Sequence Length")
    plt.title("Sequence Lengths Before and After Truncation")
    plt.legend()
    plt.grid(True)
    plt.savefig("tokens/sequence_lengths.png")
    plt.show()

def plot_attention_mask_coverage(attention_masks: torch.Tensor):
    """Plots attention mask coverage (ratio of non-padded tokens to total tokens)."""
    coverage = attention_masks.sum(dim=1).float() / attention_masks.size(1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(coverage)), coverage, color="skyblue")
    plt.xlabel("Sample Index")
    plt.ylabel("Attention Mask Coverage")
    plt.title("Attention Mask Coverage per Sample")
    plt.grid(True)
    plt.savefig("tokens/attention_mask_coverage.png")
    plt.show()

def plot_padding_efficiency(attention_masks: torch.Tensor, max_length: int):
    """Plots padding efficiency (ratio of used tokens to max length)."""
    efficiency = attention_masks.sum(dim=1).float() / max_length
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(efficiency)), efficiency, color="lightcoral")
    plt.xlabel("Sample Index")
    plt.ylabel("Padding Efficiency")
    plt.title("Padding Efficiency per Sample")
    plt.grid(True)
    plt.savefig("tokens/padding_efficiency.png")
    plt.show()

###############################################################################
# Plotting Functions (New) for Comparing All Tokenizers
###############################################################################
def plot_token_counts_comparison(token_counts: Dict[str, List[int]], sample_texts: List[str]):
    """
    Creates a grouped bar chart showing the token counts for each sample text
    across the three tokenizers: MedToken, tiktoken, SentencePiece.
    """
    # We'll assume token_counts has keys: "medtoken", "tiktoken", "sentencepiece"
    med_counts = token_counts["medtoken"]
    tiktoken_counts = token_counts["tiktoken"]
    sp_counts = token_counts["sentencepiece"]

    n_texts = len(sample_texts)
    x = np.arange(n_texts)

    # Each bar group will have 3 bars (one per tokenizer)
    width = 0.25

    plt.figure(figsize=(12, 6))
    # Shift each bar so they don't overlap
    plt.bar(x - width, med_counts, width, label="MedToken", color="royalblue")
    plt.bar(x, tiktoken_counts, width, label="tiktoken", color="seagreen")
    plt.bar(x + width, sp_counts, width, label="SentencePiece", color="darkorange")

    plt.xticks(x, [f"Txt {i}" for i in range(n_texts)], rotation=45, ha="right")
    plt.xlabel("Sample Text Index")
    plt.ylabel("Token Count")
    plt.title("Token Count Comparison: MedToken vs. tiktoken vs. SentencePiece")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.savefig("tokens/token_counts_comparison.png")
    plt.show()

def plot_speed_comparison(med_time: float, tiktoken_time: float, sp_time: float, n_iterations: int):
    """
    Plots a bar chart comparing the total time (in seconds) for tokenizing 
    the same text N times across MedToken, tiktoken, and SentencePiece.
    """
    labels = ["MedToken", "tiktoken", "SentencePiece"]
    times = [med_time, tiktoken_time, sp_time]

    plt.figure(figsize=(8, 6))
    x = np.arange(len(labels))
    barlist = plt.bar(x, times, color=["royalblue", "seagreen", "darkorange"], width=0.5)

    # Annotate each bar with the time
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f"{v:.2f}s", ha="center", fontweight="bold")

    plt.xticks(x, labels)
    plt.ylabel("Time (seconds)")
    plt.title(f"Speed Comparison over {n_iterations} Iterations")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.savefig("tokens/tokenizer_speed_comparison.png")
    plt.show()

###############################################################################
# MedToken Validation (Unchanged)
###############################################################################
def validate_medtoken(tokenizer_path: str, sample_texts: List[str], max_length: int = None):
    """Validates the MedToken tokenizer on a set of sample texts."""
    try:
        # Load the tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"MedToken file not found: {tokenizer_path}")

        logging.info(f"Loading MedToken from {tokenizer_path}")
        med_token = Tokenizer.from_file(tokenizer_path)

        # Encode texts
        logging.info("Encoding sample texts with MedToken...")
        encoded_batch = [med_token.encode(text).ids for text in sample_texts]

        # Set max_length dynamically if not provided
        if max_length is None:
            max_length = max(len(seq) for seq in encoded_batch)
            logging.info(f"Dynamic max_length set to: {max_length}")

        # Analyze sequence lengths
        original_lengths = [len(ids) for ids in encoded_batch]
        truncated_lengths = []

        # Validate sequence lengths and log truncation details
        for idx, ids in enumerate(encoded_batch):
            if len(ids) > max_length:
                logging.warning(f"Truncating sequence at index {idx} to max length {max_length}.")
                truncated_tokens = ids[max_length:]
                logging.info(f"Truncated tokens for Text {idx}: {truncated_tokens}")
                encoded_batch[idx] = ids[:max_length]
            truncated_lengths.append(len(encoded_batch[idx]))

        # Convert to tensors
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded_batch]
        attention_masks = [
            torch.tensor([1] * len(ids) + [0] * (max_length - len(ids)), dtype=torch.long)
            if len(ids) < max_length else torch.tensor([1] * max_length, dtype=torch.long)
            for ids in encoded_batch
        ]

        # Pad sequences
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0
        )
        padded_attention_masks = torch.stack(attention_masks, dim=0)

        # Log validation results
        logging.info("Validation Results (MedToken):")
        logging.info(f"Padded Input IDs Shape: {padded_input_ids.shape}")
        logging.info(f"Padded Attention Masks Shape: {padded_attention_masks.shape}")

        # Example output
        logging.info("Sample Encoded Outputs (MedToken):")
        for idx, input_ids_row in enumerate(padded_input_ids):
            logging.info(f"Text {idx}: {sample_texts[idx]}")
            logging.info(f"Input IDs: {input_ids_row.tolist()}")

        # Plot sequence lengths
        plot_sequence_lengths(original_lengths, truncated_lengths)

        # Plot attention mask coverage
        plot_attention_mask_coverage(padded_attention_masks)

        # Plot padding efficiency
        plot_padding_efficiency(padded_attention_masks, max_length)

    except Exception as e:
        logging.error(f"Error during MedToken validation: {e}")
        logging.error(traceback.format_exc())

###############################################################################
# Compare MedToken vs. tiktoken vs. SentencePiece
###############################################################################
def compare_tokenizers(
    med_token_path: str,
    sample_texts: List[str],
    openai_model: str = "gpt-3.5-turbo",
    sentencepiece_model_path: Optional[str] = None,
    n_iterations: int = 5000
):
    """
    Compare MedToken with:
      1. tiktoken (OpenAI)
      2. SentencePiece (Google)
    measuring both token counts (for *all* sample_texts) & tokenization speed (for the *first* text).
    
    Returns:
        token_counts (Dict[str, List[int]]): {
            "medtoken": [...],
            "tiktoken": [...],
            "sentencepiece": [...]
        }
        times (Dict[str, float]): {
            "medtoken": float,
            "tiktoken": float,
            "sentencepiece": float
        }
    """
    # 1) Load MedToken
    if not os.path.exists(med_token_path):
        logging.error(f"MedToken not found at {med_token_path}")
        return {}, {}

    med_token = Tokenizer.from_file(med_token_path)

    if not sample_texts:
        logging.error("No sample texts provided.")
        return {}, {}

    # Prepare data structures for comparison
    token_counts = {
        "medtoken": [],
        "tiktoken": [],
        "sentencepiece": []
    }
    times = {
        "medtoken": 0.0,
        "tiktoken": 0.0,
        "sentencepiece": 0.0
    }

    # 2) Collect token counts for *each* sample text
    # -------------------------------------------------------------------------
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(openai_model)
        except Exception as e:
            logging.error(f"Could not get tiktoken encoder for model={openai_model}: {e}")
            enc = None
    else:
        enc = None

    if spm is not None and sentencepiece_model_path and os.path.exists(sentencepiece_model_path):
        sp = spm.SentencePieceProcessor()
        sp.load(sentencepiece_model_path)
    else:
        sp = None

    for text in sample_texts:
        # MedToken
        med_ids = med_token.encode(text).ids
        token_counts["medtoken"].append(len(med_ids))

        # tiktoken
        if enc is not None:
            try:
                openai_ids = enc.encode(text)
                token_counts["tiktoken"].append(len(openai_ids))
            except Exception as e:
                logging.error(f"tiktoken error on text: {e}")
                token_counts["tiktoken"].append(0)
        else:
            token_counts["tiktoken"].append(0)

        # SentencePiece
        if sp is not None:
            sp_ids = sp.encode_as_ids(text)
            token_counts["sentencepiece"].append(len(sp_ids))
        else:
            token_counts["sentencepiece"].append(0)

    # 3) Compare speed (in repeated loops) using the first text only
    # -------------------------------------------------------------------------
    test_text = sample_texts[0]
    logging.info("")
    logging.info(f"Comparing speed with {n_iterations} iterations each on text: '{test_text}'")

    # MedToken speed
    start = time.time()
    for _ in range(n_iterations):
        _ = med_token.encode(test_text).ids
    times["medtoken"] = time.time() - start
    logging.info(f"[MedToken] took {times['medtoken']:.4f} s for {n_iterations} iterations")

    # tiktoken speed
    if enc is not None:
        start = time.time()
        for _ in range(n_iterations):
            _ = enc.encode(test_text)
        times["tiktoken"] = time.time() - start
        logging.info(f"[tiktoken] took {times['tiktoken']:.4f} s for {n_iterations} iterations")
    else:
        logging.info("[tiktoken] skipping speed test (no encoder)")

    # SentencePiece speed
    if sp is not None:
        start = time.time()
        for _ in range(n_iterations):
            _ = sp.encode_as_ids(test_text)
        times["sentencepiece"] = time.time() - start
        logging.info(f"[SentencePiece] took {times['sentencepiece']:.4f} s for {n_iterations} iterations")
    else:
        logging.info("[SentencePiece] skipping speed test (not installed or no model)")

    logging.info("Comparison done.")
    logging.info("==========================================================")

    return token_counts, times

###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    # Path to your MedToken JSON
    med_token_path = "tokenizer/tokenizer.json"

    # Sample texts for validation & comparison
    sample_texts = [
        "Patient says hello to the doctor.",
        "Patient presents with severe chest pain.",
        "Medical history includes hypertension and diabetes.",
        "The patient was administered 5mg of Atorvastatin.",
        "Symptoms include fever, cough, and shortness of breath.",
        "Lab results: WBC 12.3, RBC 4.5, Platelets 250k",
        "Patient reports allergies to penicillin and sulfa drugs.",
        "Post-operative recovery shows good wound healing."
    ]

    # 1) Validate MedToken
    validate_medtoken(med_token_path, sample_texts)

    # 2) Compare MedToken vs. tiktoken & SentencePiece
    sp_model_path = "sentencepiece.model"  # or None if you don't have it
    n_iterations = 3000  # Adjust as desired

    token_counts, times = compare_tokenizers(
        med_token_path=med_token_path,
        sample_texts=sample_texts,
        openai_model="gpt-3.5-turbo",
        sentencepiece_model_path=sp_model_path,
        n_iterations=n_iterations
    )

    # 3) Plot the comparisons
    #    A) Token counts across all sample texts
    if token_counts:
        plot_token_counts_comparison(token_counts, sample_texts)

    #    B) Speed comparison
    if times:
        plot_speed_comparison(
            med_time=times["medtoken"],
            tiktoken_time=times["tiktoken"],
            sp_time=times["sentencepiece"],
            n_iterations=n_iterations
        )
