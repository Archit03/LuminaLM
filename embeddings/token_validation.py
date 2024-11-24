import os
import re
import time
import logging
from typing import List, Dict
from dataclasses import dataclass, field
from tokenizers import Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tokenizer_validation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for the tokenizer validation."""
    required_tokens: List[str] = field(
        default_factory=lambda: ["<s>", "</s>", "<mask>"]
    )
    min_token_length: int = 1
    max_token_length: int = 100
    performance_threshold_ms: float = 100.0
    batch_size: int = 32

def preprocess_text(text: str) -> str:
    """Preprocess text to handle composite units and normalize terms."""
    composite_units = {
        r'(\d+)\s*°\s*C': r'\1°C',  # Celsius
        r'(\d+)\s*mmHg': r'\1mmHg',  # Blood pressure
        r'(\d+)\s*bpm': r'\1bpm',  # Heart rate
        r'(\d+)\s*kg/m²': r'\1kg/m²',  # BMI
    }
    for pattern, replacement in composite_units.items():
        text = re.sub(pattern, replacement, text)

    abbreviations = {
        r'\bb\.?i\.?d\b': 'twice daily',
        r'\bt\.?i\.?d\b': 'three times daily',
        r'\bq\.?d\b': 'daily',
        r'\bp\.?r\.?n\b': 'as needed',
        r'\bp\.?o\b': 'by mouth'
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load the tokenizer from a JSON file."""
    try:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        if not tokenizer_path.endswith('.json'):
            raise ValueError("Invalid file format. Expected .json file")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"Tokenizer successfully loaded from {tokenizer_path}.")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise

def validate_tokenizer_config(tokenizer: Tokenizer, config: TokenizerConfig):
    """Validate the tokenizer's configuration."""
    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in config.required_tokens if token not in vocab]
    if missing_tokens:
        logger.warning(f"Required tokens missing from tokenizer: {missing_tokens}")
    else:
        logger.info("Tokenizer configuration validation passed.")

def test_tokenizer(tokenizer: Tokenizer, test_inputs: List[str], config: TokenizerConfig) -> List[Dict]:
    """Test the tokenizer's performance on various inputs."""
    logger.info("--- Tokenizer Performance Test ---")
    results = []
    for text in test_inputs:
        preprocessed_text = preprocess_text(text)
        encoded = tokenizer.encode(preprocessed_text)
        results.append({
            "original_text": text,
            "preprocessed_text": preprocessed_text,
            "tokenized_output": encoded.tokens,
            "token_ids": encoded.ids
        })
        logger.info(f"Original Text: {text}")
        logger.info(f"Preprocessed Text: {preprocessed_text}")
        logger.info(f"Tokenized Output: {encoded.tokens}")
        logger.info(f"Token IDs: {encoded.ids}")
        logger.info("-" * 50)
    return results

def benchmark_tokenizer(tokenizer: Tokenizer, test_cases: List[str], config: TokenizerConfig):
    """Measure tokenization speed and performance."""
    logger.info("--- Batch Processing ---")
    batch_times = []
    for i in range(0, len(test_cases), config.batch_size):
        batch = test_cases[i:i + config.batch_size]
        start_time = time.time()
        for case in batch:
            tokenizer.encode(preprocess_text(case))
        batch_times.append(time.time() - start_time)

    average_time = sum(batch_times) / len(batch_times)
    max_time = max(batch_times)
    min_time = min(batch_times)

    metrics = {
        'average_time': average_time,
        'max_time': max_time,
        'min_time': min_time
    }
    logger.info(f"Average batch processing time: {average_time:.2f}s.")
    logger.info(f"Max batch processing time: {max_time:.2f}s.")
    logger.info(f"Min batch processing time: {min_time:.2f}s.")
    return metrics

def save_results_to_log_file(results, file_name="tokenizer_validation_results.log"):
    """Save validation results to a log file."""
    try:
        with open(file_name, "w", encoding="utf-8") as log_file:
            log_file.write("=== Tokenizer Validation Results ===\n\n")
            log_file.write("--- Tokenizer Performance Test ---\n")
            for result in results["performance_test"]:
                log_file.write(f"Original Text: {result['original_text']}\n")
                log_file.write(f"Preprocessed Text: {result['preprocessed_text']}\n")
                log_file.write(f"Tokenized Output: {result['tokenized_output']}\n")
                log_file.write(f"Token IDs: {result['token_ids']}\n")
                log_file.write("-" * 50 + "\n")

            log_file.write("\n--- Batch Metrics ---\n")
            for metric, value in results["batch_metrics"].items():
                log_file.write(f"{metric}: {value:.2f}\n")

        logger.info(f"Results successfully saved to {file_name}")
    except Exception as e:
        logger.error(f"Error saving results to log file: {e}")

def validate_tokenizer(tokenizer_path: str, config: TokenizerConfig):
    """Main validation function."""
    tokenizer = load_tokenizer(tokenizer_path)

    # Test inputs
    test_inputs = [
        "Patient diagnosed with Type 2 Diabetes and Hypertension.",
        "Administer 500mg Amoxicillin orally twice daily for 7 days.",
        "BP: 120/80 mmHg, HR: 72 bpm, Temp: 37.0°C.",
        "BMI: 25 kg/m², Blood glucose: 140 mg/dL.",
        "Patient reports severe chest pain radiating to the left arm.",
        "Echocardiogram revealed left ventricular hypertrophy.",
        "Prescribe 20mg Rosuvastatin for lipid management."
    ]

    # Validate configuration
    validate_tokenizer_config(tokenizer, config)

    # Test performance
    performance_results = test_tokenizer(tokenizer, test_inputs, config)

    # Benchmark
    batch_metrics = benchmark_tokenizer(tokenizer, test_inputs, config)

    # Save results
    save_results_to_log_file({
        "performance_test": performance_results,
        "batch_metrics": batch_metrics
    })


if __name__ == "__main__":
    tokenizer_path = "LuminaLM_text_tokens.json"
    config = TokenizerConfig(batch_size=32)
    validate_tokenizer(tokenizer_path, config)
