import logging
import os
import traceback
from tokenizers import Tokenizer
import tiktoken
import torch
import matplotlib.pyplot as plt
from typing import List
import time

# Logging setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("token_validation/validation.log"),
                              logging.StreamHandler()])

def plot_token_overlap(medtoken_tokens: List[List[int]], tiktoken_tokens: List[List[int]]):
    """Plots the percentage of overlapping tokens between MedToken and Tiktoken."""
    overlaps = []
    for med_tokens, tik_tokens in zip(medtoken_tokens, tiktoken_tokens):
        overlap = len(set(med_tokens) & set(tik_tokens)) / max(len(set(med_tokens)), 1) * 100
        overlaps.append(overlap)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(overlaps)), overlaps, color="limegreen", alpha=0.8, label="Token Overlap (%)")
    plt.xlabel("Sample Index")
    plt.ylabel("Overlap Percentage")
    plt.title("Token Overlap Between MedToken and Tiktoken")
    plt.xticks(range(len(overlaps)))
    plt.legend()
    plt.grid(True)
    os.makedirs("token_validation", exist_ok=True)
    plt.savefig("token_validation/medtoken_vs_tiktoken_overlap.png")
    plt.show()

def plot_token_count_difference(medtoken_lengths: List[int], tiktoken_lengths: List[int]):
    """Plots the absolute difference in token counts between MedToken and Tiktoken."""
    differences = [abs(med_len - tik_len) for med_len, tik_len in zip(medtoken_lengths, tiktoken_lengths)]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(differences)), differences, color="goldenrod", alpha=0.8, label="Token Count Difference")
    plt.xlabel("Sample Index")
    plt.ylabel("Token Count Difference")
    plt.title("Token Count Difference: MedToken vs. Tiktoken")
    plt.xticks(range(len(differences)))
    plt.legend()
    plt.grid(True)
    plt.savefig("token_validation/medtoken_vs_tiktoken_difference.png")
    plt.show()

def plot_tokenization_time(medtoken_time: float, tiktoken_time: float):
    """Plots tokenization time comparison between MedToken and Tiktoken."""
    plt.figure(figsize=(10, 6))
    plt.bar(["MedToken", "Tiktoken"], [medtoken_time, tiktoken_time], color=["royalblue", "crimson"], alpha=0.8)
    plt.ylabel("Time (seconds)")
    plt.title("Tokenization Time Comparison: MedToken vs. Tiktoken")
    for i, time_val in enumerate([medtoken_time, tiktoken_time]):
        plt.text(i, time_val + 0.001, f"{time_val:.4f}s", ha="center", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y")
    plt.savefig("token_validation/medtoken_vs_tiktoken_time.png")
    plt.show()

def plot_sequence_length_comparison(medtoken_lengths: List[int], tiktoken_lengths: List[int]):
    """Plots sequence lengths for MedToken and Tiktoken for comparison."""
    plt.figure(figsize=(10, 6))
    plt.plot(medtoken_lengths, marker="o", label="MedToken Lengths", color="blue", alpha=0.8)
    plt.plot(tiktoken_lengths, marker="x", label="Tiktoken Lengths", color="red", linestyle="--", alpha=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("Sequence Length")
    plt.title("Sequence Length Comparison: MedToken vs. Tiktoken")
    plt.xticks(range(len(medtoken_lengths)))
    plt.legend()
    plt.grid(True)
    plt.savefig("token_validation/medtoken_vs_tiktoken_lengths.png")
    plt.show()

def validate_tokenizer(tokenizer_path: str, sample_texts: List[str], max_length: int = None):
    """Validates the MedToken tokenizer and compares it with OpenAI's Tiktoken."""
    try:
        # Load the tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        logging.info(f"Loading MedToken tokenizer from {tokenizer_path}")
        medtoken_tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load OpenAI's Tiktoken GPT-3.5 encoding
        tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

        # Encode texts with MedToken
        logging.info("Encoding sample texts with MedToken...")
        start_time = time.time()
        medtoken_batch = [medtoken_tokenizer.encode(text).ids for text in sample_texts]
        medtoken_time = time.time() - start_time
        logging.info(f"MedToken time: {medtoken_time:.4f} seconds")

        # Encode texts with Tiktoken
        logging.info("Encoding sample texts with OpenAI's Tiktoken...")
        start_time = time.time()
        tiktoken_batch = [tiktoken_encoder.encode(text) for text in sample_texts]
        tiktoken_time = time.time() - start_time
        logging.info(f"Tiktoken time: {tiktoken_time:.4f} seconds")

        # Compare lengths
        medtoken_lengths = [len(ids) for ids in medtoken_batch]
        tiktoken_lengths = [len(ids) for ids in tiktoken_batch]
        logging.info("Length Comparison:")
        for i, (med_len, tik_len) in enumerate(zip(medtoken_lengths, tiktoken_lengths)):
            logging.info(f"Text {i}: MedToken Length = {med_len}, Tiktoken Length = {tik_len}")

        # Plot comparison graphs
        plot_token_overlap(medtoken_batch, tiktoken_batch)
        plot_token_count_difference(medtoken_lengths, tiktoken_lengths)
        plot_tokenization_time(medtoken_time, tiktoken_time)
        plot_sequence_length_comparison(medtoken_lengths, tiktoken_lengths)

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    tokenizer_path = "tokens/Medical_tokenizer.json"
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
