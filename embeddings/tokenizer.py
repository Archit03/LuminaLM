
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
            "<EVIDENCE>", "<CONCLUSION>", "<REFERENCE>", "<UNKNOWN>", "<MISSING>", "<ANONYMOUS>"
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
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>"))
            ]
        )
    
    def preprocess_dataset(self, dataset_name: str) -> List[str]:
        """Loads and preprocesses a dataset with error handling."""
        try:
            dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
            texts = []
            
            if "text" in dataset.column_names:
                texts.extend(dataset["text"])
            elif all(col in dataset.column_names for col in ["question", "context"]):
                texts.extend([f"{q.strip()} {c.strip()}" 
                            for q, c in zip(dataset["question"], dataset["context"])])
            elif "sentence" in dataset.column_names:
                texts.extend(dataset["sentence"])
                
            logger.info(f"Processed {len(texts)} examples from {dataset_name}")
            return texts
            
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            return []
    
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
    
    def train(self, additional_directory: Optional[str] = None) -> None:
        """Trains the tokenizer with progress tracking."""
        if not self.tokenizer:
            self.create_tokenizer()
            
        datasets = {
            "openwebtext": self.preprocess_dataset("openwebtext"),
            "medical": []
        }
        
        # Process medical datasets
        medical_datasets = ["pubmed_qa", "mednli", "i2b2_2010", "mimic_notes", "scicite"]
        for dataset_name in medical_datasets:
            datasets["medical"].extend(self.preprocess_dataset(dataset_name))
            
        # Process additional files if provided
        if additional_directory:
            datasets["additional"] = self.preprocess_files(additional_directory)
            
        # Combine all datasets
        combined_data = (datasets["openwebtext"] + 
                        datasets["medical"] + 
                        datasets.get("additional", []))
        
        if not combined_data:
            raise ValueError("No valid training data found")
            
        logger.info(f"Training tokenizer on {len(combined_data)} examples")
        self.tokenizer.train_from_iterator(combined_data, self.trainer)
        
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
    # Example usage
    tokenizer = MedicalTokenizer(vocab_size=50000)
    tokenizer.train(additional_directory="path/to/data")
    tokenizer.save()

if __name__ == "__main__":
    main()