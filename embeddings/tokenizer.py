import logging
import os
import re
import traceback
import json
import unidecode
from typing import Any, Dict, List, Optional
from enum import Enum

from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.processors import TemplateProcessing

import spacy
from spacy.lang.en import English

# Enums for various strategies and domains
class MedicalDomain(Enum):
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"

class NormalizationStrategy(Enum):
    NONE = "none"
    LEMMATIZATION = "lemmatization"
    STEMMING = "stemming"

class TokenizationStrategy(Enum):
    BPE = "bpe"
    WORDPIECE = "wordpiece"

# Constants for domain-specific tokens (example)
DOMAIN_SPECIFIC_TOKENS = {
    MedicalDomain.CARDIOLOGY.value: ["<ECG>", "<ECHO>"],
    MedicalDomain.NEUROLOGY.value: ["<EEG>", "<MRI>"],
    MedicalDomain.ONCOLOGY.value: ["<CT>", "<PET>"],
}

class EnhancedMedicalTextPreprocessor:
    def __init__(
        self,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
        min_token_length: int = 1,
        max_token_length: int = 100,
        use_medical_nlp: bool = True,
        normalization_strategy: str = NormalizationStrategy.NONE.value,
        medical_domain: Optional[str] = None,
    ):
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        self.normalization_strategy = normalization_strategy
        self.rules = self._generate_rules(custom_rules, medical_domain)
        self.nlp = self._load_nlp_model(use_medical_nlp)

    def _load_nlp_model(self, use_medical_nlp: bool):
        """
        Robust NLP model loading with comprehensive fallback strategy
        """
        try:
            if use_medical_nlp:
                try:
                    # Attempt to load medical-specific model first
                    nlp = spacy.load("en_core_sci_md")
                except OSError:
                    try:
                        # Fallback to clinical model
                        nlp = spacy.load("en_core_sci_lg")
                    except OSError:
                        logging.warning("Medical NLP models not found. Using core English model.")
                        nlp = spacy.load("en_core_web_sm")
            else:
                nlp = English()  # Minimal tokenization pipeline
            nlp.max_length = 3 * 10**7  # Set to 30 million characters
            # Optionally disable unnecessary pipeline components
            if self.normalization_strategy != NormalizationStrategy.LEMMATIZATION.value:
                nlp.disable_pipes('tagger', 'parser', 'ner')
            return nlp
        except Exception as e:
            logging.error(f"Critical NLP model loading failure: {e}")
            return None

    def _generate_rules(self, custom_rules: Optional[List[Dict[str, Any]]], medical_domain: Optional[str]):
        """
        Generate comprehensive regex-based rules
        """
        base_rules = [
            {"pattern": r"\d+", "repl": "<NUM>"},
            {"pattern": r"http\S+|www\S+|https\S+", "repl": "<URL>"},
            {"pattern": r"\b([A-Za-z])\b", "repl": ""},
            # Remove single letters
        ]

        if custom_rules:
            base_rules.extend(custom_rules)

        # Example domain-specific rules
        domain_specific_rules = {
            MedicalDomain.CARDIOLOGY.value: [
                {"pattern": r"electrocardiogram", "repl": "ECG"},
                {"pattern": r"echocardiogram", "repl": "ECHO"},
            ],
            MedicalDomain.NEUROLOGY.value: [
                {"pattern": r"electroencephalogram", "repl": "EEG"},
                {"pattern": r"magnetic resonance imaging", "repl": "MRI"},
            ],
            MedicalDomain.ONCOLOGY.value: [
                {"pattern": r"computed tomography", "repl": "CT"},
                {"pattern": r"positron emission tomography", "repl": "PET"},
            ],
            # Add more domain-specific rules as needed
        }

        # Combine base rules with domain-specific rules if domain is specified
        if medical_domain and medical_domain in domain_specific_rules:
            return base_rules + domain_specific_rules[medical_domain]

        return base_rules

    def preprocess(self, text: str) -> str:
        """
        Enhanced text preprocessing with comprehensive error handling
        """
        if not isinstance(text, str):
            logging.warning(f"Invalid input type: {type(text)}. Returning empty string.")
            return ""

        # Standardize whitespace and apply rules
        text = re.sub(r"\s+", " ", text.strip())
        for rule in self.rules:
            text = re.sub(rule["pattern"], rule["repl"], text, flags=re.IGNORECASE)

        # Split text if it's too long
        if len(text) > self.nlp.max_length:
            logging.warning(f"Text length {len(text)} exceeds max length {self.nlp.max_length}. Splitting text.")
            chunks = [text[i:i + self.nlp.max_length] for i in range(0, len(text), self.nlp.max_length)]
        else:
            chunks = [text]

        processed_tokens = []
        for chunk in chunks:
            if self.nlp:
                doc = self.nlp(chunk)

                if self.normalization_strategy == NormalizationStrategy.LEMMATIZATION.value:
                    tokens = [
                        unidecode.unidecode(token.lemma_.lower())
                        for token in doc
                        if (self.min_token_length <= len(token.text) <= self.max_token_length
                            and not token.is_punct
                            and not token.is_space)
                    ]
                elif self.normalization_strategy == NormalizationStrategy.STEMMING.value:
                    # Basic stemming using token text
                    tokens = [
                        unidecode.unidecode(token.text.lower())[:self.max_token_length]
                        for token in doc
                        if (self.min_token_length <= len(token.text) <= self.max_token_length
                            and not token.is_punct
                            and not token.is_space)
                    ]
                else:
                    # No normalization
                    tokens = [
                        unidecode.unidecode(token.text.lower())
                        for token in doc
                        if (self.min_token_length <= len(token.text) <= self.max_token_length
                            and not token.is_punct
                            and not token.is_space)
                    ]
                processed_tokens.extend(tokens)
            else:
                processed_tokens.extend(chunk.split())

        return " ".join(processed_tokens)

class MedicalTokenizer:
    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        custom_preprocessing_rules: Optional[List[Dict[str, Any]]] = None,
        local_data_path: str = "",
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
        tokenization_strategy: str = TokenizationStrategy.BPE.value,
        medical_domain: Optional[str] = None
    ):
        preprocessor_kwargs = preprocessor_kwargs or {}
        preprocessor_kwargs['medical_domain'] = medical_domain

        self.preprocessor = EnhancedMedicalTextPreprocessor(
            custom_rules=custom_preprocessing_rules,
            **preprocessor_kwargs
        )

        # Configurable tokenization strategy
        if tokenization_strategy == TokenizationStrategy.BPE.value:
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        elif tokenization_strategy == TokenizationStrategy.WORDPIECE.value:
            self.tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
        else:
            raise ValueError(f"Unsupported tokenization strategy: {tokenization_strategy}")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = self._generate_special_tokens(medical_domain)
        self.local_data_path = local_data_path
        self.tokenization_strategy = tokenization_strategy
        self.medical_domain = medical_domain

    def _generate_special_tokens(self, medical_domain: Optional[str] = None):
        """Generate comprehensive special tokens"""
        base_special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

        if medical_domain:
            domain_tokens = DOMAIN_SPECIFIC_TOKENS.get(medical_domain, [])
            return list(set(base_special_tokens + domain_tokens))

        # If no specific domain, return all possible domain tokens
        domain_tokens = [token for tokens in DOMAIN_SPECIFIC_TOKENS.values() for token in tokens]
        return list(set(base_special_tokens + domain_tokens))

    def _configure_tokenizer(self):
        """Configure tokenizer with special tokens and processing"""
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

    def train(self, datasets: List[Dict[str, Any]], output_path: str):
        """Train tokenizer on medical datasets"""
        self._configure_tokenizer()

        # Select appropriate trainer based on tokenization strategy
        if self.tokenization_strategy == TokenizationStrategy.BPE.value:
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens
            )
        else:
            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens
            )

        logging.info(f"Training tokenizer...")
        self.tokenizer.train_from_iterator(self._stream_datasets(datasets), trainer=trainer)
        self.tokenizer.save(output_path)
        logging.info(f"Tokenizer saved at {output_path}")

    def _stream_datasets(self, datasets):
        """Stream and preprocess data from datasets"""
        for dataset_info in tqdm(datasets, desc="Processing Datasets"):
            dataset_name = dataset_info["name"]
            config_name = dataset_info.get("config", None)
            split = dataset_info.get("split", "train")
            try:
                logging.info(f"Loading dataset: {dataset_name}")
                load_dataset_kwargs = {
                    "path": dataset_name,
                    "split": split,
                }
                if config_name:
                    load_dataset_kwargs["name"] = config_name

                # Include 'trust_remote_code' only if specified
                if dataset_info.get("trust_remote_code", False):
                    load_dataset_kwargs["trust_remote_code"] = True

                ds = load_dataset(**load_dataset_kwargs)

                text_column = self._get_text_column(ds)
                if not text_column:
                    logging.warning(f"No text column found in {dataset_name}. Skipping dataset.")
                    continue

                for example in tqdm(ds, desc=f"Processing {dataset_name}", leave=False):
                    text = self._extract_text(example, text_column)
                    if text:
                        preprocessed_text = self.preprocessor.preprocess(text)
                        if preprocessed_text:
                            yield preprocessed_text
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {e}")
                logging.error(traceback.format_exc())

        if self.local_data_path and os.path.exists(self.local_data_path):
            logging.info(f"Processing local data at {self.local_data_path}")
            for root, _, files in os.walk(self.local_data_path):
                for file in tqdm(files, desc="Processing local files"):
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith(".txt"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                                preprocessed_text = self.preprocessor.preprocess(text)
                                if preprocessed_text:
                                    yield preprocessed_text
                        elif file.endswith(".json"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for entry in data:
                                        text = entry.get("text", "") or entry.get("content", "")
                                        if text:
                                            preprocessed_text = self.preprocessor.preprocess(text)
                                            if preprocessed_text:
                                                yield preprocessed_text
                        elif file.endswith(".jsonl"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    entry = json.loads(line)
                                    text = entry.get("text", "") or entry.get("content", "")
                                    if text:
                                        preprocessed_text = self.preprocessor.preprocess(text)
                                        if preprocessed_text:
                                            yield preprocessed_text
                        # Add handling for other file types if needed
                    except Exception as e:
                        logging.error(f"Error reading file {file}: {e}")
                        logging.error(traceback.format_exc())

    def _extract_text(self, example, text_column):
        """Extract text from the example, handling nested structures if necessary."""
        text = example.get(text_column, "")
        if isinstance(text, dict):
            for key in ['text', 'content', 'body', 'article']:
                if key in text:
                    return text[key]
            return json.dumps(text)
        elif isinstance(text, list):
            return " ".join(text)
        return text

    def _get_text_column(self, dataset):
        possible_columns = ["text", "content", "article", "body", "sentence", "abstract"]
        if isinstance(dataset.column_names, dict):
            # For datasets with multiple splits
            for split in dataset.column_names:
                for column in possible_columns:
                    if column in dataset.column_names[split]:
                        return column
        else:
            for column in possible_columns:
                if column in dataset.column_names:
                    return column
        return None

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    datasets = [
        {"name": "ruslanmv/ai-medical-chatbot", "trust_remote_code": True},
        {"name": "qanastek/ELRC-Medical-V2", "config": "en-bg", "trust_remote_code": True},
        {"name": "qanastek/ELRC-Medical-V2", "config": "en-cs", "trust_remote_code": True},
        {"name": "qanastek/ELRC-Medical-V2", "config": "en-da", "trust_remote_code": True},
        {"name": "rungalileo/medical_transcription_40"},
        {"name": "gamino/wiki_medical_terms"},
        {"name": "medalpaca/medical_meadow_medqa"},
        {"name": "medalpaca/medical_meadow_wikidoc_patient_information"},
        {"name": "joey234/mmlu-medical_genetics-neg"},
        {"name": "joey234/mmlu-medical_genetics-verbal-neg-prepend"},
        {"name": "joey234/mmlu-medical_genetics-rule-neg"},
        {"name": "joey234/mmlu-medical_genetics"},
        {"name": "tchebonenko/MedicalTranscriptions"},
        {"name": "lavita/medical-qa-shared-task-v1-toy"},
        {"name": "lavita/medical-qa-shared-task-v1-all"},
        {"name": "lavita/medical-qa-shared-task-v1-half"},
        {"name": "lavita/medical-qa-shared-task-v1-toy-eval"},
        {"name": "hari560/medical-data"},
        {"name": "srikanthsri/medical_biological"},
        {"name": "jayantdocplix/medical_dataset"},
        {"name": "owkin/medical_knowledge_from_extracts"},
        {"name": "joey234/mmlu-medical_genetics-neg-prepend-fix"},
        {"name": "taaredikahan23/medical-llama2-1k"},
        {"name": "keivalya/MedQuad-MedicalQnADataset"},
        {"name": "Kabatubare/medical-alpaca"},
        {"name": "Kabatubare/medical"},
        {"name": "Malikeh1375/medical-question-answering-datasets", "config": "all-processed"},
        {"name": "Malikeh1375/medical-question-answering-datasets", "config": "chatdoctor_healthcaremagic"},
        {"name": "Malikeh1375/medical-question-answering-datasets", "config": "chatdoctor_icliniq"},
        {"name": "lavita/medical-qa-datasets", "config": "all-processed"},
        {"name": "lavita/medical-qa-datasets", "config": "chatdoctor-icliniq"},
        {"name": "lavita/medical-qa-datasets", "config": "chatdoctor_healthcaremagic"},
        {"name": "mamachang/medical-reasoning"},
        {"name": "Mohammed-Altaf/medical-instruction-100k"},
        {"name": "joey234/mmlu-medical_genetics-neg-prepend-verbal"},
        {"name": "hpe-ai/medical-cases-classification-tutorial"},
        {"name": "bhargavi909/Medical_Transcriptions_upsampled"},
        {"name": "lamhieu/medical_medqa_dialogue_en"},
        {"name": "bala1524/Medical-QA-Mistral7B-Finetuning"},
        {"name": "pubmed_qa", "config": "pqa_artificial"},
        {"name": "scicite"},
        {"name": "openwebtext"}

        # Add any other datasets you wish to include
    ]

    local_data_path = "path_to_your_local_data"  # Update this path accordingly

    tokenizer = MedicalTokenizer(
        vocab_size=50257,
        min_frequency=2,
        local_data_path=local_data_path,
        preprocessor_kwargs={
            "min_token_length": 2,
            "max_token_length": 40,
            "use_medical_nlp": True,
            "normalization_strategy": NormalizationStrategy.LEMMATIZATION.value,
        },
        tokenization_strategy=TokenizationStrategy.BPE.value,
        medical_domain=None  # Set to specific domain if needed
    )
    output_path = "medical_tokenizer.json"
    tokenizer.train(datasets, output_path)

if __name__ == "__main__":
    main()
