import logging
import os
import traceback
from tokenizers import Tokenizer
import torch
import matplotlib.pyplot as plt
from typing import List, Dict

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("tokens/validation.log"),
                              logging.StreamHandler()])

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

def validate_tokenizer(tokenizer_path: str, sample_texts: List[str], max_length: int = None):
    """Validates the tokenizer on a set of sample texts."""
    try:
        # Load the tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Encode texts
        logging.info("Encoding sample texts...")
        encoded_batch = [
            tokenizer.encode(text).ids for text in sample_texts
        ]

        # Set max_length dynamically based on the longest sequence if not provided
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
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        padded_attention_masks = torch.stack(attention_masks, dim=0)

        # Log validation results
        logging.info("Validation Results:")
        logging.info(f"Padded Input IDs Shape: {padded_input_ids.shape}")
        logging.info(f"Padded Attention Masks Shape: {padded_attention_masks.shape}")

        # Example output
        logging.info("Sample Encoded Outputs:")
        for idx, input_ids in enumerate(padded_input_ids):
            logging.info(f"Text {idx}: {sample_texts[idx]}")
            logging.info(f"Input IDs: {input_ids.tolist()}")

        # Plot sequence lengths
        plot_sequence_lengths(original_lengths, truncated_lengths)

        # Plot attention mask coverage
        plot_attention_mask_coverage(padded_attention_masks)

        # Plot padding efficiency
        plot_padding_efficiency(padded_attention_masks, max_length)

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    tokenizer_path = "tokens/tokenizer.json"
    sample_texts = [
        "Patient presents with severe chest pain.",
        "Medical history includes hypertension and diabetes.",
        "The patient was administered 5mg of Atorvastatin.",
        "Symptoms include fever, cough, and shortness of breath.",
        "Lab results: WBC 12.3, RBC 4.5, Platelets 250k",
        "Patient reports allergies to penicillin and sulfa drugs.",
        "Post-operative recovery shows good wound healing."
    ]

    validate_tokenizer(tokenizer_path, sample_texts)
