import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
import os
import pandas as pd
from typing import List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Loads and preprocesses multiple datasets for tokenizer training."""

    def __init__(self):
        self.datasets = [
            {"name": "Malikeh1375/medical-question-answering-datasets", "config": None},
            {"name": "lavita/medical-qa-datasets", "config": None},
            {"name": "Karmukilan/Malikeh1375_medical-question-answering-datasets", "config": None},
            {"name": "balu1235/medical_question_answering_datasets", "config": None},
            {"name": "community-datasets/swedish_medical_ner", "config": None},
            {"name": "bio-datasets/re-medical-annotations", "config": None},
            {"name": "Lei-USYD/datasets_medical", "config": None},
            {"name": "xDAN-datasets/medical_meadow_wikidoc_10k", "config": None},
            {"name": "xDAN-datasets/medical_meadow_wikidoc_patient_information_6k", "config": None},
            {"name": "xDAN-datasets/medical_meadow_mediqa_2k", "config": None},
            {"name": "kd4ser/medical-question-answering-datasets", "config": None},
            {"name": "blue-blue/medical_dataset_shards", "config": None},
            {"name": "qiaojin/PubMedQA", "config": "pqa_artificial"},
            {"name": "qiaojin/PubMedQA", "config": "pqa_labeled"},
            {"name": "qiaojin/PubMedQA", "config": "pqa_unlabeled"},
            {"name": "bigbio/scicite", "config": "scicite_bigbio_text"},
            {"name": "ruslanmv/ai-medical-dataset", "config": None},
        ]

    def load_all_datasets(self) -> list:
        """Loads all datasets and returns combined text data."""
        all_texts = []

        for dataset_info in self.datasets:
            name = dataset_info["name"]
            config = dataset_info.get("config")

            try:
                if config:
                    dataset = load_dataset(name, config, split="train")
                else:
                    dataset = load_dataset(name, split="train")

                # Extract text fields dynamically
                texts = self.extract_texts(dataset)
                all_texts.extend(texts)

                logger.info(f"Loaded {len(texts)} examples from {name} ({config or 'default'})")

            except Exception as e:
                logger.error(f"Error loading {name} ({config or 'default'}): {str(e)}")

        logger.info(f"Total combined examples: {len(all_texts)}")
        return all_texts

    @staticmethod
    def extract_texts(dataset) -> list:
        """Extracts relevant text fields from a dataset."""
        texts = []

        if "text" in dataset.column_names:
            texts.extend(dataset["text"])
        elif all(col in dataset.column_names for col in ["question", "answer"]):
            texts.extend([f"{entry['question']} {entry['answer']}" for entry in dataset])
        elif "sentence" in dataset.column_names:
            texts.extend(dataset["sentence"])
        elif all(col in dataset.column_names for col in ["context", "response"]):
            texts.extend([f"{entry['context']} {entry['response']}" for entry in dataset])
        elif "description" in dataset.column_names:
            texts.extend(dataset["description"])
        else:
            logger.warning("No recognized text columns found.")
        
        return texts


class MedicalTokenizer:
    """A specialized tokenizer for medical text processing."""

    def __init__(self, vocab_size: int = 50000, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or self._get_default_special_tokens()
        self.tokenizer = None
        self.trainer = None

    @staticmethod
    def _get_default_special_tokens() -> List[str]:
        """Returns the default set of special tokens for medical text processing."""
        return [
            "<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eot>", "<bos>", "<eos>",
            # Add additional medical-specific tokens here
        ]

    def create_tokenizer(self) -> None:
        """Initializes the tokenizer with BPE and special tokens."""
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2  # Minimum frequency for a token to be included
        )

        # Add post-processor for handling special tokens
        self.tokenizer.train_from_iterator(["dummy text for training"], self.trainer)  # Dummy training to register special tokens
        special_tokens = [
            ("<s>", self.tokenizer.token_to_id("<s>")),
            ("</s>", self.tokenizer.token_to_id("</s>"))
        ]

        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=special_tokens
        )

    def train(self, additional_directory: Optional[str] = None) -> int:
        """Trains the tokenizer and counts total tokens."""
        if not self.tokenizer:
            self.create_tokenizer()

        # Load all datasets dynamically
        loader = DatasetLoader()
        datasets = {
            "all_datasets": loader.load_all_datasets(),
            "additional": []
        }

        # Process additional files if provided
        if additional_directory:
            datasets["additional"] = self.preprocess_files(additional_directory)

        # Combine all datasets
        combined_data = datasets["all_datasets"] + datasets.get("additional", [])

        if not combined_data:
            raise ValueError("No valid training data found")

        logger.info(f"Training tokenizer on {len(combined_data)} examples")
        self.tokenizer.train_from_iterator(combined_data, self.trainer)

        # Count tokens
        total_tokens = sum(len(self.tokenizer.encode(text).ids) for text in combined_data)
        logger.info(f"Total tokens processed: {total_tokens}")

        return total_tokens

    def preprocess_files(self, directory: str) -> List[str]:
        """Processes files from a directory with enhanced error handling."""
        texts = []
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.error(f"Directory {directory} does not exist")
            return texts

        for file_path in directory_path.rglob("*"):
            try:
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                    text = " ".join(df.astype(str).values.flatten())
                    texts.append(text)
                elif file_path.suffix == ".txt":
                    text = file_path.read_text(encoding='utf-8')
                    texts.append(text)

                logger.info(f"Processed {file_path.name}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

        return texts

    def save(self, path: str = "LuminaLM_text_tokens.json") -> None:
        """Saves the tokenizer with error handling."""
        try:
            self.tokenizer.save(path)
            logger.info(f"Tokenizer saved to {path}")
        except Exception as e:
            logger.error(f"Error saving tokenizer: {str(e)}")

    @classmethod
    def load(cls, path: str = "LuminaLM_text_tokens.json") -> 'MedicalTokenizer':
        """Loads a saved tokenizer."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance


def main():
    tokenizer = MedicalTokenizer(vocab_size=50000)
    total_tokens = tokenizer.train(additional_directory="LuminaLM/Data")
    tokenizer.save()
    logger.info(f"Tokenizer training completed. Total tokens processed: {total_tokens}")
    print(f"Total tokens processed{total_tokens} tokens")


if __name__ == "__main__":
    main()
