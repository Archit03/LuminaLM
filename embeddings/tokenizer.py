import os
import re
import json
import csv
import time
import tracemalloc
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Optional, Dict, Any, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset
from tqdm import tqdm

# Configure logging with timestamps
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'tokenizer_training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates and analyzes medical datasets."""

    def __init__(self):
        self.medical_terms = self._load_medical_terms()
        self.stats = {
            "total_examples": 0,
            "medical_content": 0,
            "non_medical_content": 0,
            "invalid_format": 0,
            "dataset_statistics": {}
        }

    def _load_medical_terms(self) -> set:
        """Loads medical terms or creates a default set."""
        # For simplicity, using a predefined set of medical terms
        return {
            'patient', 'doctor', 'hospital', 'diagnosis', 'treatment',
            'symptoms', 'disease', 'medical', 'clinical', 'health',
            'medicine', 'prescription', 'surgery', 'therapy', 'blood',
            'imaging', 'laboratory', 'acute', 'chronic', 'oncology',
            'cardiology', 'neurology', 'orthopedic', 'radiology', 'pharmacy'
        }

    def is_medical_content(self, text: str) -> bool:
        """Determines if text contains medical content."""
        if not isinstance(text, str):
            return False

        text_lower = text.lower()

        # Check for medical terms
        medical_term_count = sum(1 for term in self.medical_terms if term in text_lower)

        # Check for medical patterns
        medical_patterns = [
            r'\d+\s*(?:mg|ml|mcg|g|kg|mmol|mmhg)',  # Measurements
            r'diagnosis\s*(?:of|:)',  # Diagnosis patterns
            r'(?:presenting|presents?\s+with)',  # Clinical presentation
            r'(?:prescribed|medication|dosage)',  # Medication patterns
            r'(?:lab|test)\s*results?',  # Lab results
        ]

        pattern_matches = any(re.search(pattern, text_lower) for pattern in medical_patterns)

        # Consider text medical if it has multiple medical terms or matches patterns
        return medical_term_count >= 2 or pattern_matches

    def validate_example(self, text: str) -> bool:
        """Validates a single text example."""
        self.stats["total_examples"] += 1

        if not text or not isinstance(text, str):
            self.stats["invalid_format"] += 1
            return False

        if self.is_medical_content(text):
            self.stats["medical_content"] += 1
            return True

        self.stats["non_medical_content"] += 1
        return False

class DatasetLoader:
    """Loads and preprocesses multiple datasets for tokenizer training."""

    def __init__(self, batch_size: int = 1000, local_data_paths: Optional[List[str]] = None):
        self.batch_size = batch_size
        self.validator = DatasetValidator()
        self.datasets = self._get_dataset_configs()
        self.local_data_paths = local_data_paths or []

    def _get_dataset_configs(self) -> List[Dict[str, Any]]:
        """Returns configuration for all medical datasets."""
        return [
            {"name": "ruslanmv/ai-medical-chatbot", "config": None},
            {"name": "qanastek/ELRC-Medical-V2", "config": "en-bg"},
            {"name": "qanastek/ELRC-Medical-V2", "config": "en-cs"},
            {"name": "qanastek/ELRC-Medical-V2", "config": "en-da"},
            {"name": "rungalileo/medical_transcription_40", "config": None},
            {"name": "gamino/wiki_medical_terms", "config": None},
            {"name": "medalpaca/medical_meadow_medqa", "config": None},
            {"name": "medalpaca/medical_meadow_wikidoc_patient_information", "config": None},
            {"name": "joey234/mmlu-medical_genetics-neg", "config": None},
            {"name": "joey234/mmlu-medical_genetics-verbal-neg-prepend", "config": None},
            {"name": "joey234/mmlu-medical_genetics-rule-neg", "config": None},
            {"name": "joey234/mmlu-medical_genetics", "config": None},
            {"name": "tchebonenko/MedicalTranscriptions", "config": None},
            {"name": "lavita/medical-qa-shared-task-v1-toy", "config": None},
            {"name": "lavita/medical-qa-shared-task-v1-all", "config": None},
            {"name": "lavita/medical-qa-shared-task-v1-half", "config": None},
            {"name": "lavita/medical-qa-shared-task-v1-toy-eval", "config": None},
            {"name": "hari560/medical-data", "config": None},
            {"name": "srikanthsri/medical_biological", "config": None},
            {"name": "jayantdocplix/medical_dataset", "config": None},
            {"name": "owkin/medical_knowledge_from_extracts", "config": None},
            {"name": "joey234/mmlu-medical_genetics-neg-prepend-fix", "config": None},
            {"name": "taaredikahan23/medical-llama2-1k", "config": None},
            {"name": "keivalya/MedQuad-MedicalQnADataset", "config": None},
            {"name": "Kabatubare/medical-alpaca", "config": None},
            {"name": "Kabatubare/medical", "config": None},
            {"name": "Malikeh1375/medical-question-answering-datasets", "config": "all-processed"},
            {"name": "Malikeh1375/medical-question-answering-datasets", "config": "chatdoctor_healthcaremagic"},
            {"name": "Malikeh1375/medical-question-answering-datasets", "config": "chatdoctor_icliniq"},
            {"name": "lavita/medical-qa-datasets", "config": "all-processed"},
            {"name": "lavita/medical-qa-datasets", "config": "chatdoctor-icliniq"},
            {"name": "lavita/medical-qa-datasets", "config": "chatdoctor_healthcaremagic"},
            {"name": "mamachang/medical-reasoning", "config": None},
            {"name": "Mohammed-Altaf/medical-instruction-100k", "config": None},
            {"name": "joey234/mmlu-medical_genetics-neg-prepend-verbal", "config": None},
            {"name": "hpe-ai/medical-cases-classification-tutorial", "config": None},
            {"name": "bhargavi909/Medical_Transcriptions_upsampled", "config": None},
            {"name": "lamhieu/medical_medqa_dialogue_en", "config": None},
            {"name": "bala1524/Medical-QA-Mistral7B-Finetuning", "config": None}
        ]

    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text."""
        if not isinstance(text, str):
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Normalize medical abbreviations
        abbreviations = {
            r'\bb\.?i\.?d\b': 'twice daily',
            r'\bt\.?i\.?d\b': 'three times daily',
            r'\bq\.?d\b': 'daily',
            r'\bp\.?r\.?n\b': 'as needed',
            r'\bp\.?o\b': 'by mouth',
            # Add more medical abbreviations if needed
        }

        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Normalize measurements
        text = re.sub(r'(\d+)\s*(mg|ml|g|kg|mcg)', r'\1 \2', text, flags=re.IGNORECASE)

        return text

    def _extract_text_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """Extracts text from a data item (e.g., a JSON object or CSV row)."""
        # Assuming that the text content is under the key 'text', 'content', or similar
        possible_keys = ['text', 'content', 'message', 'data']
        for key in possible_keys:
            if key in item and isinstance(item[key], str):
                return item[key]
        # If the item doesn't contain expected keys, return None
        return None

    def _load_files_from_directory(self, directory_path: str) -> List[str]:
        """Recursively collects all file paths from the directory."""
        file_paths = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
        return file_paths

    def load_local_data(self) -> Iterator[str]:
        """Loads data from local files and directories."""
        all_file_paths = []
        for path in self.local_data_paths:
            if os.path.isdir(path):
                logger.info(f"Loading local data from directory: {path}")
                files_in_dir = self._load_files_from_directory(path)
                all_file_paths.extend(files_in_dir)
            elif os.path.isfile(path):
                all_file_paths.append(path)
            else:
                logger.warning(f"Local data path not found: {path}")
                continue

        for file_path in all_file_paths:
            logger.info(f"Processing local file: {file_path}")
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            text = line.strip()
                            if text:
                                text = self._preprocess_text(text)
                                if self.validator.validate_example(text):
                                    yield text
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            text = self._extract_text_from_item(item)
                            if text:
                                text = self._preprocess_text(text)
                                if self.validator.validate_example(text):
                                    yield text
                elif file_path.endswith('.csv'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            text = self._extract_text_from_item(row)
                            if text:
                                text = self._preprocess_text(text)
                                if self.validator.validate_example(text):
                                    yield text
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
            except Exception as e:
                logger.error(f"Error loading local data from {file_path}: {str(e)}")

    def load_dataset_safely(self, dataset_info: Dict[str, Any]) -> Iterator[str]:
        """Safely loads a single dataset with validation and error handling."""
        name = dataset_info["name"]
        config = dataset_info.get("config")
        dataset_name = f"{name} ({config or 'default'})"

        try:
            dataset = load_dataset(
                name,
                config,
                split="train",
                streaming=True
            )

            logger.info(f"Loading dataset: {dataset_name}")

            for example in dataset:
                text = self._extract_text_from_item(example)
                if text:
                    text = self._preprocess_text(text)
                    if self.validator.validate_example(text):
                        yield text

        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {str(e)}")

    def load_all_datasets(self) -> Iterator[str]:
        """Loads all datasets and local data."""
        # First, load datasets from Hugging Face
        for dataset_info in self.datasets:
            yield from self.load_dataset_safely(dataset_info)

        # Then, load local data
        if self.local_data_paths:
            yield from self.load_local_data()

class MedicalTokenizer:
    """Enhanced tokenizer for medical text processing."""

    def __init__(
        self,
        vocab_size: int = 50000,
        special_tokens: Optional[List[str]] = None,
        batch_size: int = 1000,
        local_data_paths: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or self._get_default_special_tokens()
        self.tokenizer = None
        self.trainer = None
        self.batch_size = batch_size
        self.local_data_paths = local_data_paths or []
        self.performance_stats = {
            "training_time": 0,
            "processed_examples": 0,
            "vocab_coverage": {},
            "token_frequencies": Counter(),
            "memory_usage_mb": 0
        }
        self.training_start_time = None
        self.training_end_time = None
        self.training_texts_sample = []

    @staticmethod
    def _get_default_special_tokens() -> List[str]:
        """Returns enhanced set of special tokens for medical text processing."""
        base_tokens = [
            "<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>",
            "<eot>", "<bos>", "<eos>", "<SYM>", "<DIAG>", "<PROC>", "<TREAT>",
            "<MED>", "<DOSAGE>", "<FREQ>", "<ROUTE>", "<LAB>", "<VAL>",
            "<IMAGING>", "<BLOOD>", "<VITALS>", "<DISEASE>", "<CONDITION>",
            "<ALLERGY>", "<FAMILY_HISTORY>", "<SOCIAL_HISTORY>", "<ORG>",
            "<BODY_PART>", "<TISSUE>", "<SYSTEM>", "<MUSCLE>", "<NORMAL>",
            "<ABNORMAL>", "<SEVERE>", "<MODERATE>", "<MILD>", "<STABLE>",
            "<IMPROVING>", "<WORSENING>", "<SURGERY>", "<NONINVASIVE>",
            "<INVASIVE>", "<THERAPY>", "<TRANSPLANT>", "<BIOPSY>", "<MRI>",
            "<CT>", "<XRAY>", "<ULTRASOUND>", "<RESULT>", "<POSITIVE>",
            "<NEGATIVE>", "<DATE>", "<DURATION>", "<TIMESTAMP>", "<AGE>",
            "<GENDER>", "<WEIGHT>", "<HEIGHT>", "<PATIENT_ID>", "<CONSENT>",
            "<HIPAA>", "<ICD_CODE>", "<CPT_CODE>", "<GLUCOSE>", "<BP>",
            "<HR>", "<O2_SAT>", "<TEMP>", "<RBC>", "<WBC>", "<PLATELET>",
            "<COVID19>", "<HYPERTENSION>", "<DIABETES>", "<CANCER>", "<STROKE>",
            "<CARDIAC>", "<PRESCRIPTION>", "<GENERIC_DRUG>", "<BRAND_DRUG>",
            "<DOSAGE_FORM>", "<GENE>", "<MUTATION>", "<DNA>", "<RNA>",
            "<PROTEIN>", "<GENOTYPE>", "<SNP>", "<SEQ>", "<MG>", "<ML>",
            "<L>", "<MOL>", "<IU>", "<STUDY>", "<TRIAL>", "<EVIDENCE>",
            "<CONCLUSION>", "<REFERENCE>", "<UNKNOWN>", "<MISSING>", "<ANONYMOUS>"
        ]

        # Additional medical special tokens
        additional_tokens = [
            "<MMOL_L>", "<MG_DL>", "<KG_M2>", "<CELSIUS>", "<FAHRENHEIT>",
            "<ENDOSCOPY>", "<COLONOSCOPY>", "<BIOPSY>", "<SURGERY_TYPE>",
            "<CBC>", "<CMP>", "<HBA1C>", "<LIPID_PANEL>",
            "<CARDIOLOGY>", "<NEUROLOGY>", "<ONCOLOGY>", "<PEDIATRICS>",
            "<ACUTE>", "<CHRONIC>", "<REMISSION>", "<RELAPSE>",
            "<DOSAGE_ADJUSTMENT>", "<SIDE_EFFECT>", "<CONTRAINDICATION>",
            "<INSURANCE>", "<BILLING_CODE>", "<REFERRAL>"
        ]

        return base_tokens + additional_tokens

    def create_tokenizer(self) -> None:
        """Initializes an enhanced tokenizer with medical-specific configurations."""
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        self.tokenizer.add_special_tokens(self.special_tokens)

        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet(),
            continuing_subword_prefix="##"
        )

        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>"))
            ]
        )

    def train(self) -> int:
        """Trains the tokenizer with enhanced monitoring and validation."""
        if not self.tokenizer:
            self.create_tokenizer()

        tracemalloc.start()
        self.training_start_time = datetime.now()
        loader = DatasetLoader(batch_size=self.batch_size, local_data_paths=self.local_data_paths)

        processed_texts = 0

        def data_generator():
            nonlocal processed_texts
            for text in loader.load_all_datasets():
                processed_texts += 1
                self.performance_stats["processed_examples"] += 1
                if len(self.training_texts_sample) < 1000:
                    self.training_texts_sample.append(text)
                if processed_texts % 1000 == 0:
                    elapsed = (datetime.now() - self.training_start_time).total_seconds()
                    logger.info(f"Processed {processed_texts} texts in {elapsed:.2f} seconds")
                yield text

        logger.info("Starting tokenizer training...")
        self.tokenizer.train_from_iterator(
            data_generator(),
            self.trainer
        )

        self.training_end_time = datetime.now()
        # Calculate training time
        self.performance_stats["training_time"] = (self.training_end_time - self.training_start_time).total_seconds()
        current, peak = tracemalloc.get_traced_memory()
        self.performance_stats["memory_usage_mb"] = peak / 10**6
        tracemalloc.stop()
        logger.info(f"Peak memory usage: {self.performance_stats['memory_usage_mb']:.2f} MB")

        # Validate and analyze
        self._validate_tokenizer()
        self.compute_token_length_statistics()
        self.validate_special_tokens()
        self.analyze_vocabulary_coverage(self._get_validation_samples())
        self.benchmark_tokenizer_performance(self._get_validation_samples())
        self.analyze_token_frequencies()
        self.save_performance_metrics()
        self.save_metadata()

        return self.tokenizer.get_vocab_size()

    def _get_validation_samples(self) -> List[str]:
        """Returns a set of validation samples."""
        return [
            "Patient presents with acute myocardial infarction.",
            "Prescribed 500mg Amoxicillin twice daily for 7 days.",
            "MRI reveals L4-L5 disc herniation.",
            "Blood pressure 120/80 mmHg, heart rate 72 bpm.",
            "Family history positive for type 2 diabetes mellitus.",
            "Post-operative recovery following total knee arthroplasty.",
            "CBC shows WBC 12.3, RBC 4.5, Platelets 250.",
            "Patient reports severe chest pain radiating to left arm.",
            "Drug dosage adjusted due to renal insufficiency.",
            "Neurological examination reveals intact cranial nerves."
        ]

    def _validate_tokenizer(self) -> None:
        """Validates the trained tokenizer with sample medical texts."""
        sample_texts = self._get_validation_samples()

        logger.info("Validating tokenizer with sample texts...")
        for text in sample_texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded.ids)
            if decoded != text:
                logger.warning(f"Decoded text does not match original.\nOriginal: {text}\nDecoded: {decoded}")
            else:
                logger.info(f"Validation successful for text: {text}")
            logger.debug(f"Tokens: {encoded.tokens}\n")

    def compute_token_length_statistics(self) -> None:
        """Computes statistics on token lengths."""
        token_lengths = [len(token) for token in self.tokenizer.get_vocab().keys()]
        mean_length = np.mean(token_lengths)
        median_length = np.median(token_lengths)
        max_length = np.max(token_lengths)
        logger.info(f"Token length statistics - Mean: {mean_length:.2f}, Median: {median_length}, Max: {max_length}")

    def validate_special_tokens(self) -> None:
        """Validates the presence of special tokens in the vocabulary."""
        missing_tokens = [token for token in self.special_tokens if token not in self.tokenizer.get_vocab()]
        if missing_tokens:
            logger.warning(f"Missing special tokens: {missing_tokens}")
        else:
            logger.info("All special tokens are present in the vocabulary.")

    def analyze_vocabulary_coverage(self, sample_texts: List[str]) -> None:
        """Analyzes how well the vocabulary covers the sample texts."""
        total_tokens = 0
        oov_tokens = 0

        for text in sample_texts:
            encoded = self.tokenizer.encode(text)
            total_tokens += len(encoded.tokens)
            oov_tokens += encoded.tokens.count('<unk>')

        coverage = (total_tokens - oov_tokens) / total_tokens * 100
        logger.info(f"Vocabulary coverage: {coverage:.2f}%")
        self.performance_stats["vocab_coverage"] = coverage

    def benchmark_tokenizer_performance(self, sample_texts: List[str]) -> None:
        """Benchmarks tokenizer performance on sample texts."""
        start_time = time.time()
        for text in sample_texts:
            _ = self.tokenizer.encode(text)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Tokenization time for {len(sample_texts)} texts: {total_time:.2f} seconds")
        logger.info(f"Average time per text: {total_time / len(sample_texts):.4f} seconds")
        self.performance_stats["tokenization_time_per_text"] = total_time / len(sample_texts)

    def analyze_token_frequencies(self) -> None:
        """Analyzes token frequencies in the vocabulary."""
        token_counts = Counter()
        for text in self.training_texts_sample:
            encoded = self.tokenizer.encode(text)
            token_counts.update(encoded.tokens)

        most_common = token_counts.most_common(10)
        least_common = token_counts.most_common()[:-11:-1]
        logger.info(f"Most common tokens: {most_common}")
        logger.info(f"Least common tokens: {least_common}")

        # Update performance stats
        self.performance_stats["token_frequencies"] = dict(most_common)

    def save_performance_metrics(self, path: str = "performance_metrics.json") -> None:
        """Saves performance metrics to a JSON file."""
        metrics = {
            "training_time_seconds": self.performance_stats["training_time"],
            "processed_examples": self.performance_stats["processed_examples"],
            "vocab_size": self.tokenizer.get_vocab_size(),
            "memory_usage_mb": self.performance_stats.get("memory_usage_mb", None),
            "vocab_coverage": self.performance_stats.get("vocab_coverage", None),
            "tokenization_time_per_text": self.performance_stats.get("tokenization_time_per_text", None),
            "token_frequencies": self.performance_stats.get("token_frequencies", None),
            # Add other metrics as needed
        }
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Performance metrics saved to {path}")

    def save_metadata(self, path: str = "training_metadata.json") -> None:
        """Saves metadata about the training process."""
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "batch_size": self.batch_size,
            "local_data_paths": self.local_data_paths,
            "training_start_time": self.training_start_time.isoformat(),
            "training_end_time": self.training_end_time.isoformat(),
            "total_training_time_seconds": self.performance_stats["training_time"],
            "processed_examples": self.performance_stats["processed_examples"],
            # Add more metadata as needed
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Training metadata saved to {path}")

    def save(self, path: str = "LuminaLM_text_tokens.json") -> None:
        """Saves the tokenizer with error handling and validation."""
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            self.tokenizer.save(str(save_path))
            logger.info(f"Tokenizer successfully saved to {save_path}")

            # Save vocabulary statistics
            vocab = self.tokenizer.get_vocab()
            logger.info(f"Vocabulary size: {len(vocab)}")
            logger.info(f"Special tokens included: {[token for token in self.special_tokens if token in vocab]}")

            # Verify the saved file
            if save_path.exists():
                file_size = save_path.stat().st_size
                logger.info(f"Verified saved file. Size: {file_size / 1024:.2f} KB")
            else:
                raise FileNotFoundError("Tokenizer file was not saved successfully")

        except Exception as e:
            logger.error(f"Error saving tokenizer: {str(e)}")
            raise

def main():
    try:
        # Specify paths to your local data directories or files
        local_data_paths = [
            'LuminaLM/Data',
            # Add more paths if you have multiple directories or files
        ]

        # Initialize tokenizer with configuration and local data paths
        tokenizer = MedicalTokenizer(
            vocab_size=50000,
            batch_size=1000,
            local_data_paths=local_data_paths
        )

        # Train the tokenizer
        vocab_size = tokenizer.train()

        # Save the trained tokenizer
        tokenizer.save(path=f"LuminaLM_text_tokens_{timestamp}.json")

        logger.info(f"Tokenizer training completed successfully. Final vocabulary size: {vocab_size}")
    except Exception as e:
        logger.exception(f"An error occurred during tokenizer training: {str(e)}")

if __name__ == "__main__":
    main()
