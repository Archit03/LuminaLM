import logging
from tokenizers import Tokenizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# Path to the trained tokenizer
trained_tokenizer_path = "tokenizer/tokenizer.json"

# Load the trained tokenizer
def load_tokenizer(path):
    try:
        logging.info("Loading tokenizer...")
        tokenizer = Tokenizer.from_file(path)
        logging.info("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise

# Load dataset from a file
def load_dataset(file_path):
    try:
        logging.info("Loading dataset...")
        with tqdm(total=Path(file_path).stat().st_size, desc="Reading File", unit="bytes") as pbar:
            with open(file_path, "r", encoding="utf-8") as file:
                lines = []
                for line in file:
                    pbar.update(len(line.encode('utf-8')))
                    stripped_line = line.strip()
                    if stripped_line:
                        lines.append(stripped_line)
        logging.info(f"Dataset loaded successfully with {len(lines)} lines.")
        return lines
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Analyze token coverage
def analyze_token_coverage_parallel(texts, tokenizer):
    total_tokens = 0
    oov_tokens = 0
    oov_token_set = set()

    def process_text(text):
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        oov = [token for token in tokens if token not in tokenizer.get_vocab()]
        return len(tokens), len(oov), set(oov)

    with tqdm(total=len(texts), desc="Analyzing Token Coverage", unit="text") as pbar:
        with ThreadPoolExecutor() as executor:
            results = []
            for result in executor.map(process_text, texts):
                results.append(result)
                pbar.update(1)  # Increment progress after each document

    for tokens, oov_count, oov_set in results:
        total_tokens += tokens
        oov_tokens += oov_count
        oov_token_set.update(oov_set)

    coverage = ((total_tokens - oov_tokens) / total_tokens) * 100
    logging.info(f"Token Coverage: {coverage:.2f}%")
    logging.info(f"Total Tokens: {total_tokens}, OOV Tokens: {oov_tokens}")
    if oov_tokens > 0:
        logging.info(f"OOV Tokens (examples): {list(oov_token_set)[:10]}")
    return coverage, list(oov_token_set)

# Inspect tokenization quality
def inspect_tokenization_quality_parallel(texts, tokenizer):
    def process_text(text):
        encoded = tokenizer.encode(text)
        return text, encoded.tokens

    with tqdm(total=len(texts), desc="Inspecting Tokenization", unit="text") as pbar:
        with ThreadPoolExecutor() as executor:
            results = []
            for result in executor.map(process_text, texts):
                results.append(result)
                pbar.update(1)  # Increment progress after each document

    for original_text, tokenized in results:
        logging.info(f"Original Text: {original_text}")
        logging.info(f"Tokenized: {tokenized}")
        logging.info("-" * 50)

# Main function
def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load the tokenizer
    tokenizer = load_tokenizer(trained_tokenizer_path)

    # Load dataset from file
    dataset_file = r"C:\Users\ASUS\Downloads\large_text_dataset_100k.txt"  # Replace with your dataset file path
    sample_texts = load_dataset(dataset_file)

    # Analyze token coverage
    logging.info("Analyzing token coverage...")
    coverage, oov_tokens = analyze_token_coverage_parallel(sample_texts, tokenizer)

    # Provide recommendations
    if coverage < 98:
        logging.warning(f"Coverage is below 98%. Consider increasing vocabulary size.")
    elif coverage > 99:
        logging.info(f"Excellent token coverage. No adjustments needed.")
    else:
        logging.info(f"Token coverage is acceptable.")

    # Inspect tokenization quality
    logging.info("Inspecting tokenization quality...")
    inspect_tokenization_quality_parallel(sample_texts, tokenizer)

if __name__ == "__main__":
    main()
