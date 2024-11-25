import os
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

# Configure logging with timestamps
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tokenizer_training_{timestamp}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and preprocesses datasets for tokenizer training."""

    def __init__(self, local_data_path: str):
        self.local_data_path = local_data_path
        self._validate_local_data_path()

    def _validate_local_data_path(self):
        """Validates the local directory path."""
        if not os.path.exists(self.local_data_path):
            raise FileNotFoundError(f"Local data directory does not exist: {self.local_data_path}")
        if not os.path.isdir(self.local_data_path):
            raise NotADirectoryError(f"Path exists but is not a directory: {self.local_data_path}")
        logger.info(f"Local data directory validated: {self.local_data_path}")

    def _sanitize_dataset_name(self, name: str) -> str:
        """Sanitizes dataset names to remove invalid characters."""
        return re.sub(r'[^\w./_-]', '', name.strip())

    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text to normalize units and terms."""
        if not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text.strip())

        composite_units = {
            r'(\d+)\s*°\s*C': r'\1°C',
            r'(\d+)\s*mmHg': r'\1mmHg',
            r'(\d+)\s*kg/m²': r'\1kg/m²',
            r'(\d+)\s*bpm': r'\1bpm',
        }
        for pattern, replacement in composite_units.items():
            text = re.sub(pattern, replacement, text)

        abbreviations = {
            r'\bb\.?i\.?d\b': 'twice daily',
            r'\bt\.?i\.?d\b': 'three times daily',
            r'\bq\.?d\b': 'daily',
            r'\bp\.?r\.?n\b': 'as needed',
            r'\bp\.?o\b': 'by mouth',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def load_dataset_safely(self, dataset_info: Dict[str, Any]) -> Iterator[str]:
        """Safely loads a single Hugging Face dataset."""
        name = self._sanitize_dataset_name(dataset_info["name"])
        config = dataset_info.get("config")
        dataset_name = f"{name} ({config or 'default'})"

        try:
            dataset = load_dataset(name, config, split="train")
            logger.info(f"Loading dataset: {dataset_name}")
            for example in dataset:
                text = example.get("text") or example.get("content") or ""
                if text:
                    yield self._preprocess_text(text)
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {str(e)}")

    def load_all_data(self, datasets: List[Dict[str, Any]]) -> Iterator[str]:
        """Loads all datasets and local data."""
        for dataset_info in datasets:
            yield from self.load_dataset_safely(dataset_info)

        for file_name in os.listdir(self.local_data_path):
            file_path = os.path.join(self.local_data_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            yield self._preprocess_text(line.strip())
                except Exception as e:
                    logger.error(f"Error reading file {file_name}: {str(e)}")


class MedicalTokenizer:
    """Tokenizer for medical text."""

    def __init__(self, vocab_size: int = 50000, local_data_path: str = ""):
        self.vocab_size = vocab_size
        self.local_data_path = local_data_path
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.special_tokens = self._get_special_tokens()

    def _get_special_tokens(self) -> List[str]:
        """Returns a list of special tokens."""
        return ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

    def configure_tokenizer(self):
        """Configures the tokenizer."""
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[("<s>", self.tokenizer.token_to_id("<s>")), ("</s>", self.tokenizer.token_to_id("</s>"))]
        )

    def train(self, datasets: List[Dict[str, Any]]):
        """Trains the tokenizer."""
        self.configure_tokenizer()
        loader = DatasetLoader(local_data_path=self.local_data_path)

        def data_generator():
            for text in loader.load_all_data(datasets):
                yield text

        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        logger.info("Starting tokenizer training...")
        self.tokenizer.train_from_iterator(data_generator(), trainer)
        logger.info("Tokenizer training complete.")

    def save(self, output_path: str):
        """Saves the tokenizer."""
        self.tokenizer.save(output_path)
        logger.info(f"Tokenizer saved to {output_path}")


def main():
    try:
        datasets = [
            {"name": "rungalileo/medical_transcription_40"},
            {"name": "gamino/wiki_medical_terms"},
            {"name": "medalpaca/medical_meadow_medqa"},
            {"name": "joey234/mmlu-medical_genetics-neg"},
            {"name": "joey234/mmlu-medical_genetics-verbal-neg-prepend"},
            {"name": "joey234/mmlu-medical_genetics-rule-neg"},
            {"name": "tchebonenko/MedicalTranscriptions"},
            {"name": "srikanthsri/medical_biological"},
            {"name": "openwebtext"}
        ]
        local_data_path = r"C:\Users\ASUS\Desktop\LuminaLM\Data"

        tokenizer = MedicalTokenizer(local_data_path=local_data_path)
        tokenizer.train(datasets)
        tokenizer.save("LuminaLM_text_tokens.json")

    except Exception as e:
        logger.error(f"Error during tokenizer training: {str(e)}")


if __name__ == "__main__":
    main()
