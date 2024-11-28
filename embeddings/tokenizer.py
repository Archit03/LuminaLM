import os
import re
import logging
import json
import csv
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import spacy
import unidecode
import traceback  # Added for detailed error logging

# Suppress symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class MedicalTextPreprocessor:
    """Comprehensive preprocessor for medical text with configurable rules."""

    def __init__(
        self,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
        min_token_length: int = 2,
        max_token_length: int = 30,
        use_medical_nlp: bool = True,
    ):
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length

        # Default medical-specific normalization rules
        self.default_rules = [
            {"pattern": r"(\d+)\s*°\s*C", "repl": r"\1°C"},
            {"pattern": r"(\d+)\s*mmHg", "repl": r"\1mmHg"},
            {"pattern": r"(\d+)\s*kg/m²", "repl": r"\1kg/m²"},
            {"pattern": r"(\d+)\s*bpm", "repl": r"\1bpm"},
            {"pattern": r"\bb\.?i\.?d\b", "repl": "twice daily", "flags": re.IGNORECASE},
            {"pattern": r"\bt\.?i\.?d\b", "repl": "three times daily", "flags": re.IGNORECASE},
            {"pattern": r"\bq\.?d\b", "repl": "daily", "flags": re.IGNORECASE},
            {"pattern": r"\bp\.?r\.?n\b", "repl": "as needed", "flags": re.IGNORECASE},
            {"pattern": r"\bp\.?o\b", "repl": "by mouth", "flags": re.IGNORECASE},
        ]
        self.rules = self.default_rules + (custom_rules or [])
        self.nlp = None
        if use_medical_nlp:
            try:
                self.nlp = spacy.load("en_core_sci_md")
            except OSError:
                logging.warning("Medical NLP model not found. Falling back to standard preprocessing.")
                logging.info("You can install it using: python -m spacy download en_core_sci_md")

    def _apply_regex_rules(self, text: str) -> str:
        for rule in self.rules:
            text = re.sub(rule["pattern"], rule["repl"], text, flags=rule.get("flags", 0))
        return text

    def _medical_nlp_normalize(self, text: str) -> str:
        if not self.nlp:
            return text
        doc = self.nlp(text)
        normalized_tokens = []
        for token in doc:
            lemma = token.lemma_
            normalized = unidecode.unidecode(lemma.lower())
            if (
                self.min_token_length <= len(normalized) <= self.max_token_length
                and not token.is_punct
                and not token.is_space
            ):
                normalized_tokens.append(normalized)
        return " ".join(normalized_tokens)

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s+", " ", text.strip())
        text = self._apply_regex_rules(text)
        text = self._medical_nlp_normalize(text)
        return text


class MedicalTokenizer:
    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        custom_preprocessing_rules: Optional[List[Dict[str, Any]]] = None,
        local_data_path: str = "",
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.local_data_path = local_data_path
        preprocessor_kwargs = preprocessor_kwargs or {}
        if custom_preprocessing_rules:
            preprocessor_kwargs["custom_rules"] = custom_preprocessing_rules
        self.preprocessor = MedicalTextPreprocessor(**preprocessor_kwargs)
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

    def _configure_tokenizer(self):
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )

    def _stream_datasets(self, datasets):
        for dataset_info in tqdm(datasets, desc="Processing Datasets"):
            dataset_name = dataset_info["name"]
            config_name = dataset_info.get("config", None)
            split = dataset_info.get("split", "train")
            try:
                logging.info(f"Loading dataset: {dataset_name}")
                if config_name:
                    ds = load_dataset(
                        dataset_name, config_name, split=split, trust_remote_code=True
                    )
                else:
                    ds = load_dataset(
                        dataset_name, split=split, trust_remote_code=True
                    )

                text_column = self._get_text_column(ds)
                if not text_column:
                    logging.warning(f"No text column found in {dataset_name}. Skipping dataset.")
                    continue

                # Handle datasets with nested structures
                for example in tqdm(ds, desc=f"Processing {dataset_name}", leave=False):
                    text = self._extract_text(example, text_column)
                    if text:
                        yield self.preprocessor.preprocess(text)
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {e}")
                logging.error(traceback.format_exc())

        if self.local_data_path and os.path.exists(self.local_data_path):
            logging.info(f"Processing local dataset at {self.local_data_path}")
            for root, _, files in os.walk(self.local_data_path):
                for file in tqdm(files, desc="Processing local files"):
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith(".txt"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                                yield self.preprocessor.preprocess(text)
                        elif file.endswith(".json"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for entry in data:
                                        text = entry.get("text", "") or entry.get("content", "")
                                        if text:
                                            yield self.preprocessor.preprocess(text)
                            # Handle JSONL files
                        elif file.endswith(".json"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    entry = json.loads(line)
                                    text = entry.get("text", "") or entry.get("content", "")
                                    if text:
                                        yield self.preprocessor.preprocess(text)
                        elif file.endswith(".csv"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    text = row.get("text", "") or row.get("content", "")
                                    if text:
                                        yield self.preprocessor.preprocess(text)
                    except Exception as e:
                        logging.error(f"Error reading file {file}: {e}")
                        logging.error(traceback.format_exc())

    def _extract_text(self, example, text_column):
        """Extract text from the example, handling nested structures if necessary."""
        text = example.get(text_column, "")
        if isinstance(text, dict):
            # Handle cases where text is nested inside another dict
            for key in ['text', 'content', 'body', 'article']:
                if key in text:
                    return text[key]
            return json.dumps(text)  # Fallback to converting the dict to a string
        return text

    def _get_text_column(self, dataset):
        possible_columns = ["text", "content", "article", "body", "sentence", "abstract"]
        for column in possible_columns:
            if column in dataset.column_names:
                return column
        # Check if dataset has multiple columns in each split
        if isinstance(dataset.column_names, dict):
            for split in dataset.column_names:
                for column in possible_columns:
                    if column in dataset.column_names[split]:
                        return column
        return None

    def train(self, datasets: List[Dict[str, Any]], output_path: Optional[str] = None):
        self._configure_tokenizer()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency,
        )
        logging.info("Starting tokenizer training...")
        self.tokenizer.train_from_iterator(
            self._stream_datasets(datasets),
            trainer=trainer,
        )
        logging.info("Tokenizer training complete.")
        if output_path:
            self.save(output_path)

    def save(self, path: str):
        self.tokenizer.save(path)
        logging.info(f"Tokenizer saved to {path}")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    custom_rules = [
        {"pattern": r"\bBP\b", "repl": "blood pressure", "flags": re.IGNORECASE},
        {"pattern": r"\bhr\b", "repl": "heart rate", "flags": re.IGNORECASE},
    ]
    datasets = [
        {"name": "pubmed_qa", "config": "pqa_artificial"},
        {"name": "scicite"},
        {"name": "openwebtext"},
    ]
    local_data_path = "C:\\Users\\ASUS\\Desktop\\LuminaLM\\Data"
    tokenizer = MedicalTokenizer(
        vocab_size=60000,
        min_frequency=3,
        custom_preprocessing_rules=custom_rules,
        local_data_path=local_data_path,
        preprocessor_kwargs={
            "min_token_length": 2,
            "max_token_length": 40,
            "use_medical_nlp": True,
        },
    )
    tokenizer.train(datasets, output_path="medical_tokenizer.json")


if __name__ == "__main__":
    main()
