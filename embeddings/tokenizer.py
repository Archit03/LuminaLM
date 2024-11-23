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
        return ["<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>", "<eot>", "<bos>", "<eos>",
            "<SYM>", "<DIAG>", "<PROC>", "<TREAT>", "<MED>", "<DOSAGE>", "<FREQ>", "<ROUTE>", "<LAB>", "<VAL>",
            "<IMAGING>", "<BLOOD>", "<VITALS>", "<DISEASE>", "<CONDITION>", "<ALLERGY>", "<FAMILY_HISTORY>",
            "<SOCIAL_HISTORY>", "<ORG>", "<BODY_PART>", "<TISSUE>", "<SYSTEM>", "<MUSCLE>", "<NORMAL>", "<ABNORMAL>",
            "<SEVERE>", "<MODERATE>", "<MILD>", "<STABLE>", "<IMPROVING>", "<WORSENING>", "<SURGERY>", "<NONINVASIVE>",
            "<INVASIVE>", "<THERAPY>", "<TRANSPLANT>", "<BIOPSY>", "<MRI>", "<CT>", "<XRAY>", "<ULTRASOUND>", "<RESULT>",
            "<POSITIVE>", "<NEGATIVE>", "<DATE>", "<DURATION>", "<TIMESTAMP>", "<AGE>", "<GENDER>", "<WEIGHT>", "<HEIGHT>",
            "<PATIENT_ID>", "<CONSENT>", "<HIPAA>", "<ICD_CODE>", "<CPT_CODE>", "<GLUCOSE>", "<BP>", "<HR>", "<O2_SAT>",
            "<TEMP>", "<RBC>", "<WBC>", "<PLATELET>", "<COVID19>", "<HYPERTENSION>", "<DIABETES>", "<CANCER>", "<STROKE>",
            "<CARDIAC>", "<PRESCRIPTION>", "<GENERIC_DRUG>", "<BRAND_DRUG>", "<DOSAGE_FORM>", "<GENE>", "<MUTATION>", "<DNA>",
            "<RNA>", "<PROTEIN>", "<GENOTYPE>", "<SNP>", "<SEQ>", "<MG>", "<ML>", "<L>", "<MOL>", "<IU>", "<STUDY>", "<TRIAL>",
            "<EVIDENCE>", "<CONCLUSION>", "<REFERENCE>", "<UNKNOWN>", "<MISSING>", "<ANONYMOUS>"]

    def create_tokenizer(self) -> None:
        """Initializes the tokenizer with BPE and special tokens."""
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2
        )

        self.tokenizer.train_from_iterator(["dummy text for training"], self.trainer)

        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>"))
            ]
        )

    def train(self) -> int:
        """Trains the tokenizer and counts total tokens."""
        if not self.tokenizer:
            self.create_tokenizer()

        loader = DatasetLoader()
        combined_data = loader.load_all_datasets()

        logger.info(f"Training tokenizer on {len(combined_data)} examples")
        self.tokenizer.train_from_iterator(combined_data, self.trainer)

        total_tokens = sum(len(self.tokenizer.encode(text).ids) for text in combined_data)
        logger.info(f"Total tokens processed: {total_tokens}")

        return total_tokens

    def save(self, path: str = "LuminaLM_text_tokens.json") -> None:
        """Saves the tokenizer."""
        self.tokenizer.save(path)
        logger.info(f"Tokenizer saved to {path}")


def main():
    tokenizer = MedicalTokenizer(vocab_size=50000)
    total_tokens = tokenizer.train()
    tokenizer.save()
    logger.info(f"Tokenizer training completed. Total tokens processed: {total_tokens}")


if __name__ == "__main__":
    main()
