import os
import re
import logging
import json
import traceback
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import unidecode

# Expanded Tokenization and NLP Libraries
import spacy
from spacy.lang.en import English
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

# Expanded Medical Domains
class MedicalDomain:
    GENERAL = "GENERAL"
    ONCOLOGY = "ONCOLOGY"
    CARDIOLOGY = "CARDIOLOGY"
    NEUROLOGY = "NEUROLOGY"
    PEDIATRICS = "PEDIATRICS"
    PSYCHIATRY = "PSYCHIATRY"
    RADIOLOGY = "RADIOLOGY"
    DERMATOLOGY = "DERMATOLOGY"
    ENDOCRINOLOGY = "ENDOCRINOLOGY"
    NEPHROLOGY = "NEPHROLOGY"
    PULMONOLOGY = "PULMONOLOGY"
    GASTROENTEROLOGY = "GASTROENTEROLOGY"
    INFECTIOUS_DISEASE = "INFECTIOUS_DISEASE"
    HEMATOLOGY = "HEMATOLOGY"

# Domain-Specific Tokens
DOMAIN_SPECIFIC_TOKENS = {
    MedicalDomain.GENERAL: [
        "<disease>", "</disease>", "<symptom>", "</symptom>",
        "<treatment>", "</treatment>", "<medication>", "</medication>"
    ],
    MedicalDomain.ONCOLOGY: [
        "<tumor>", "</tumor>", "<metastasis>", "</metastasis>",
        "<staging>", "</staging>", "<chemotherapy>", "</chemotherapy>"
    ],
    MedicalDomain.CARDIOLOGY: [
        "<ECG>", "</ECG>", "<stent>", "</stent>",
        "<cholesterol>", "</cholesterol>", "<cardiac_marker>", "</cardiac_marker>"
    ],
    MedicalDomain.NEUROLOGY: [
        "<brain_region>", "</brain_region>", "<seizure>", "</seizure>",
        "<neurodegeneration>", "</neurodegeneration>", "<cognition>", "</cognition>"
    ],
    MedicalDomain.PEDIATRICS: [
        "<growth>", "</growth>", "<vaccine>", "</vaccine>",
        "<development>", "</development>", "<pediatric_disease>", "</pediatric_disease>"
    ],
    MedicalDomain.PSYCHIATRY: [
        "<mental_health>", "</mental_health>", "<anxiety>", "</anxiety>",
        "<depression>", "</depression>", "<psychosis>", "</psychosis>"
    ],
    MedicalDomain.RADIOLOGY: [
        "<xray>", "</xray>", "<MRI>", "</MRI>",
        "<CT_scan>", "</CT_scan>", "<ultrasound>", "</ultrasound>"
    ],
    MedicalDomain.DERMATOLOGY: [
        "<skin_lesion>", "</skin_lesion>", "<rash>", "</rash>",
        "<melanoma>", "</melanoma>", "<eczema>", "</eczema>"
    ],
    MedicalDomain.ENDOCRINOLOGY: [
        "<hormone>", "</hormone>", "<thyroid>", "</thyroid>",
        "<insulin>", "</insulin>", "<diabetes>", "</diabetes>"
    ],
    MedicalDomain.NEPHROLOGY: [
        "<kidney>", "</kidney>", "<dialysis>", "</dialysis>",
        "<renal_function>", "</renal_function>", "<electrolyte>", "</electrolyte>"
    ],
    MedicalDomain.PULMONOLOGY: [
        "<lung>", "</lung>", "<respiratory>", "</respiratory>",
        "<breathing>", "</breathing>", "<pulmonary_marker>", "</pulmonary_marker>"
    ],
    MedicalDomain.GASTROENTEROLOGY: [
        "<digestive_system>", "</digestive_system>", "<GI_tract>", "</GI_tract>",
        "<intestinal_marker>", "</intestinal_marker>", "<gastric_condition>", "</gastric_condition>"
    ],
    MedicalDomain.INFECTIOUS_DISEASE: [
        "<pathogen>", "</pathogen>", "<infection>", "</infection>",
        "<immune_response>", "</immune_response>", "<viral_marker>", "</viral_marker>"
    ],
    MedicalDomain.HEMATOLOGY: [
        "<blood_cell>", "</blood_cell>", "<platelet>", "</platelet>",
        "<hemoglobin>", "</hemoglobin>", "<blood_disorder>", "</blood_disorder>"
    ]
}

# Combine all special tokens from all domains
ALL_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
for tokens in DOMAIN_SPECIFIC_TOKENS.values():
    ALL_SPECIAL_TOKENS.extend(tokens)
# Remove duplicates
ALL_SPECIAL_TOKENS = list(set(ALL_SPECIAL_TOKENS))

class NormalizationStrategy:
    """Enum-like class for normalization strategies"""
    LEMMATIZATION = "lemmatization"
    STEMMING = "stemming"
    NONE = "none"

class TokenizationStrategy:
    """Enum-like class for tokenization strategies"""
    BPE = "bpe"
    WORDPIECE = "wordpiece"

class EnhancedMedicalTextPreprocessor:
    """Advanced Medical Text Preprocessor with Enhanced Configurability"""
    def __init__(
        self,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
        min_token_length: int = 2,
        max_token_length: int = 30,
        normalization_strategy: str = NormalizationStrategy.LEMMATIZATION,
        use_medical_nlp: bool = True,
        medical_domain: Optional[str] = None
    ):
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        self.normalization_strategy = normalization_strategy
        self.medical_domain = medical_domain
        self.rules = self._get_comprehensive_domain_rules(medical_domain) + (custom_rules or [])
        self.nlp = self._load_nlp_model(use_medical_nlp)

    def _load_nlp_model(self, use_medical_nlp: bool):
        """
        Robust NLP model loading with comprehensive fallback strategy
        """
        try:
            if use_medical_nlp:
                try:
                    # Attempt to load medical-specific model first
                    return spacy.load("en_core_sci_md")
                except OSError:
                    try:
                        # Fallback to clinical model
                        return spacy.load("en_core_sci_lg")
                    except OSError:
                        logging.warning("Medical NLP models not found. Using core English model.")
                        return spacy.load("en_core_web_sm")
            else:
                return English()  # Minimal tokenization pipeline
        except Exception as e:
            logging.error(f"Critical NLP model loading failure: {e}")
            return None

    def _get_comprehensive_domain_rules(self, medical_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Expanded domain-specific normalization rules with more comprehensive coverage
        """
        base_rules = [
            # Standard medical notation normalization
            {"pattern": r"(\d+)\s*mmHg", "repl": r"\1 mmHg"},
            {"pattern": r"(\d+)\s*°\s*C", "repl": r"\1°C"},
            {"pattern": r"(\d+)\s*mg/dL", "repl": r"\1 mg_per_dL"},

            # Medical abbreviation standardization
            {"pattern": r"\bBP\b", "repl": "blood_pressure"},
            {"pattern": r"\bHR\b", "repl": "heart_rate"},
            {"pattern": r"\bBMI\b", "repl": "body_mass_index"},
        ]

        # Specialized rules for specific domains
        domain_specific_rules = {
            MedicalDomain.ONCOLOGY: [
                {"pattern": r"stage\s*([IVAB]+)", "repl": r"stage_\1"},
                {"pattern": r"metastatic", "repl": "advanced_cancer"},
            ],
            MedicalDomain.CARDIOLOGY: [
                {"pattern": r"coronary artery disease", "repl": "CAD"},
                {"pattern": r"myocardial infarction", "repl": "heart_attack"},
            ],
            MedicalDomain.NEUROLOGY: [
                {"pattern": r"parkinson's disease", "repl": "PD"},
                {"pattern": r"alzheimers", "repl": "AD"},
            ]
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

        # Advanced text processing
        if self.nlp:
            doc = self.nlp(text)

            if self.normalization_strategy == NormalizationStrategy.LEMMATIZATION:
                processed_tokens = [
                    unidecode.unidecode(token.lemma_.lower())
                    for token in doc
                    if (self.min_token_length <= len(token.text) <= self.max_token_length
                        and not token.is_punct
                        and not token.is_space)
                ]
            elif self.normalization_strategy == NormalizationStrategy.STEMMING:
                # Basic stemming using token text
                processed_tokens = [
                    unidecode.unidecode(token.text.lower())[:self.max_token_length]
                    for token in doc
                    if (self.min_token_length <= len(token.text) <= self.max_token_length
                        and not token.is_punct
                        and not token.is_space)
                ]
            else:
                # No normalization
                processed_tokens = [
                    unidecode.unidecode(token.text.lower())
                    for token in doc
                    if (self.min_token_length <= len(token.text) <= self.max_token_length
                        and not token.is_punct
                        and not token.is_space)
                ]

            return " ".join(processed_tokens)

        return text

class MedicalTokenizer:
    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        custom_preprocessing_rules: Optional[List[Dict[str, Any]]] = None,
        local_data_path: str = "",
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
        tokenization_strategy: str = TokenizationStrategy.BPE,
        medical_domain: Optional[str] = None
    ):
        preprocessor_kwargs = preprocessor_kwargs or {}
        preprocessor_kwargs['medical_domain'] = medical_domain

        self.preprocessor = EnhancedMedicalTextPreprocessor(
            custom_rules=custom_preprocessing_rules,
            **preprocessor_kwargs
        )

        # Configurable tokenization strategy
        if tokenization_strategy == TokenizationStrategy.BPE:
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        elif tokenization_strategy == TokenizationStrategy.WORDPIECE:
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
        if self.tokenization_strategy == TokenizationStrategy.BPE:
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
        {"name": "pubmed_qa", "config": "pqa_artificial"},
        {"name": "scicite"},
        {"name": "openwebtext"}
    ]
    local_data_path = "path_to_your_local_data"  # Update this path accordingly

    tokenizer = MedicalTokenizer(
        vocab_size=50000,
        min_frequency=2,
        local_data_path=local_data_path,
        preprocessor_kwargs={
            "min_token_length": 2,
            "max_token_length": 40,
            "use_medical_nlp": True,
            "normalization_strategy": NormalizationStrategy.LEMMATIZATION,
        },
        tokenization_strategy=TokenizationStrategy.BPE,
        medical_domain=None  # Set to specific domain if needed
    )
    output_path = "medical_tokenizer.json"
    tokenizer.train(datasets, output_path)

if __name__ == "__main__":
    main()
