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
        logging.FileHandler(f'tokenizer_training_{timestamp}.log'),
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
            error_message = f"Error: Local data directory does not exist: {self.local_data_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        if not os.path.isdir(self.local_data_path):
            error_message = f"Error: Path exists but is not a directory: {self.local_data_path}"
            logger.error(error_message)
            raise NotADirectoryError(error_message)
        logger.info(f"Local data directory validated: {self.local_data_path}")

    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text to normalize units and terms."""
        if not isinstance(text, str):
            return ""

        # Normalize whitespace and remove excessive spaces
        text = re.sub(r'\s+', ' ', text.strip())

        # Handle composite units (merge units into single terms)
        composite_units = {
            r'(\d+)\s*°\s*C': r'\1°C',    # Celsius
            r'(\d+)\s*mmHg': r'\1mmHg',   # Blood pressure
            r'(\d+)\s*kg/m²': r'\1kg/m²', # BMI
            r'(\d+)\s*bpm': r'\1bpm',     # Heart rate
        }
        for pattern, replacement in composite_units.items():
            text = re.sub(pattern, replacement, text)

        # Normalize common abbreviations
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

    def _extract_text_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        """Extracts text from a data item."""
        keys = ['text', 'content', 'message', 'data']
        for key in keys:
            if key in item and isinstance(item[key], str):
                return item[key]
        return None

    def load_local_data(self) -> Iterator[str]:
        """Loads data from a single local directory."""
        for file_name in os.listdir(self.local_data_path):
            file_path = os.path.join(self.local_data_path, file_name)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            yield self._preprocess_text(line.strip())
                except Exception as e:
                    logger.error(f"Error reading file {file_name}: {str(e)}")
            else:
                logger.warning(f"Skipping unsupported file: {file_name}")

    def load_dataset_safely(self, dataset_info: Dict[str, Any]) -> Iterator[str]:
        """Safely loads a single Hugging Face dataset."""
        name = dataset_info["name"]
        config = dataset_info.get("config")
        dataset_name = f"{name} ({config or 'default'})"

        try:
            dataset = load_dataset(
                name,
                config,
                split="train",
                streaming=False
            )

            logger.info(f"Loading dataset: {dataset_name}")
            for example in dataset:
                text = self._extract_text_from_item(example)
                if text:
                    yield self._preprocess_text(text)

        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {str(e)}")

    def load_all_data(self, datasets: List[Dict[str, Any]]) -> Iterator[str]:
        """Loads all datasets and local data."""
        for dataset_info in datasets:
            yield from self.load_dataset_safely(dataset_info)

        yield from self.load_local_data()


class MedicalTokenizer:
    """Tokenizer for medical text."""

    def __init__(self, vocab_size: int = 50000, local_data_path: str = ""):
        self.vocab_size = vocab_size
        self.local_data_path = local_data_path
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.special_tokens = self._get_special_tokens()

    def _get_special_tokens(self) -> List[str]:
        """Returns a list of special tokens."""
        return [
    "<pad>", "<unk>", "<s>", "</s>", "<cls>", "<sep>", "<mask>","<|en|>" ,"<eot>", "<bos>", "<eos>",
    "<SYM>", "<DIAG>", "<PROC>", "<TREAT>", "<MED>", "<DOSAGE>", "<FREQ>", "<ROUTE>",
    "<LAB>", "<VAL>", "<IMAGING>", "<BLOOD>", "<VITALS>", "<DISEASE>", "<CONDITION>",
    "<ALLERGY>", "<FAMILY_HISTORY>", "<SOCIAL_HISTORY>", "<ORG>", "<BODY_PART>", "<TISSUE>",
    "<SYSTEM>", "<MUSCLE>", "<NORMAL>", "<ABNORMAL>", "<SEVERE>", "<MODERATE>", "<MILD>",
    "<STABLE>", "<IMPROVING>", "<WORSENING>", "<SURGERY>", "<NONINVASIVE>", "<INVASIVE>",
    "<THERAPY>", "<TRANSPLANT>", "<BIOPSY>", "<MRI>", "<CT>", "<XRAY>", "<ULTRASOUND>",
    "<RESULT>", "<POSITIVE>", "<NEGATIVE>", "<DATE>", "<DURATION>", "<TIMESTAMP>", "<AGE>",
    "<GENDER>", "<WEIGHT>", "<HEIGHT>", "<PATIENT_ID>", "<CONSENT>", "<HIPAA>", "<ICD_CODE>",
    "<CPT_CODE>", "<GLUCOSE>", "<BP>", "<HR>", "<O2_SAT>", "<TEMP>", "<RBC>", "<WBC>",
    "<PLATELET>", "<COVID19>", "<HYPERTENSION>", "<DIABETES>", "<CANCER>", "<STROKE>",
    "<CARDIAC>", "<PRESCRIPTION>", "<GENERIC_DRUG>", "<BRAND_DRUG>", "<DOSAGE_FORM>",
    "<GENE>", "<MUTATION>", "<DNA>", "<RNA>", "<PROTEIN>", "<GENOTYPE>", "<SNP>", "<SEQ>",
    "<MG>", "<ML>", "<L>", "<MOL>", "<IU>", "<STUDY>", "<TRIAL>", "<EVIDENCE>",
    "<CONCLUSION>", "<REFERENCE>", "<UNKNOWN>", "<MISSING>", "<ANONYMOUS>", "<MMOL_L>",
    "<MG_DL>", "<KG_M2>", "<CELSIUS>", "<FAHRENHEIT>", "<ENDOSCOPY>", "<COLONOSCOPY>",
    "<HBA1C>", "<LIPID_PANEL>", "<ULCER>", "<VIRAL>", "<BACTERIAL>"
    "<VIRAL_LOAD>", "<BACTERIAL_LOAD>", "<HEPATITIS_B_LOAD>", "<HEPATITIS_C_LOAD>",
    "<HEPATITIS_D_LOAD>", "<HEPATITIS_E_LOAD>", "<HEPATITIS_A_LOAD>", "<HEPATITIS_K_LOAD>" 
    ]

    def configure_tokenizer(self):
        """Configures the tokenizer with a BPE model and special tokens."""
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
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

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens
        )

        logger.info("Starting tokenizer training...")
        self.tokenizer.train_from_iterator(data_generator(), trainer)
        logger.info("Tokenizer training complete.")

    def save(self, output_path: str):
        """Saves the tokenizer."""
        self.tokenizer.save(output_path)
        logger.info(f"Tokenizer saved to {output_path}")


def main():
    try:
        # Only one Hugging Face dataset and one local directory
        datasets = [{"name": "rungalileo/medical_transcription_40"}, 
                   {"name": "qanastek/ELRC-Medical-V2", "config": "en-bg"},
                   {"name": "qanastek/ELRC-Medical-V2", "config": "en-cs"},
                   {"name": "rungalileo/medical_transcription_40"}, 
                   {"name" : "gamino/wiki_medical_terms"}, 
                   {"name":"medalpaca/medical_meadow_medqa"}]
        local_data_path = r"C:\Users\ASUS\Desktop\LuminaLM\Data"

        tokenizer = MedicalTokenizer(local_data_path=local_data_path)
        tokenizer.train(datasets)
        tokenizer.save(f"LuminaLM_text_tokens.json")

    except FileNotFoundError as e:
        logger.error(f"Local directory error: {str(e)}")
    except NotADirectoryError as e:
        logger.error(f"Invalid directory error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
