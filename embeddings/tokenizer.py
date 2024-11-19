import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from datasets import load_dataset
import os
import pandas as pd

# Create and configure a tokenizer
def create_tokenizer(vocab_size=50000, special_tokens=None):
    """
    Initializes the tokenizer with Byte Pair Encoding (BPE) and special tokens.

    Parameters:
        vocab_size (int): Size of the vocabulary.
        special_tokens (list): List of special tokens to include.

    Returns:
        tokenizer (Tokenizer): Configured tokenizer object.
        trainer (BpeTrainer): Trainer object for the tokenizer.
    """
    if special_tokens is None:
        special_tokens = [
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

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer, trainer

# Preprocess OpenWebText data
def preprocess_openwebtext():
    """
    Loads and preprocesses OpenWebText dataset.

    Returns:
        list: A list of text data from the OpenWebText dataset.
    """
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    return [item["text"] for item in dataset]

# Preprocess Hugging Face medical datasets
def preprocess_medical_datasets():
    """
    Loads and preprocesses various Hugging Face medical datasets.

    Returns:
        list: Combined text data from all medical datasets.
    """
    datasets_to_load = [
        "pubmed_qa", "mednli", "i2b2_2010", "mimic_notes", "scicite"
    ]

    all_texts = []
    for dataset_name in datasets_to_load:
        dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        column_names = dataset.column_names
        if "text" in column_names:
            all_texts.extend(dataset["text"])
        elif "question" in column_names and "context" in column_names:
            all_texts.extend([f"{q} {c}" for q, c in zip(dataset["question"], dataset["context"])])
        elif "sentence" in column_names:
            all_texts.extend(dataset["sentence"])
    return all_texts

# Preprocess files from a directory
def preprocess_files_from_directory(directory):
    """
    Loads and preprocesses text and CSV files from the given directory.

    Parameters:
        directory (str): Path to the directory containing text or CSV files.

    Returns:
        list: A list of preprocessed text data.
    """
    all_texts = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".csv"):
                all_texts.append(preprocess_csv(file_path))
            elif file_path.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_texts.append(f.read())
    return all_texts

# Preprocess CSV files
def preprocess_csv(file):
    """
    Loads and preprocesses a CSV file.

    Parameters:
        file (str): Path to the CSV file.

    Returns:
        str: Combined text from the CSV file.
    """
    df = pd.read_csv(file)
    return " ".join(df.astype(str).values.flatten())

# Train the tokenizer
def train_tokenizer(tokenizer, trainer, additional_directory=None):
    """
    Trains the tokenizer on OpenWebText, Hugging Face medical datasets, and optional local data.

    Parameters:
        tokenizer (Tokenizer): The tokenizer to train.
        trainer (BpeTrainer): The trainer object for the tokenizer.
        additional_directory (str): Path to additional local data (optional).
    """
    openwebtext_data = preprocess_openwebtext()
    medical_data = preprocess_medical_datasets()
    additional_texts = preprocess_files_from_directory(additional_directory) if additional_directory else []

    combined_data = openwebtext_data + medical_data + additional_texts
    if combined_data:
        tokenizer.train_from_iterator(combined_data, trainer)
    else:
        raise ValueError("No valid text data found for training.")

# Save the tokenizer
def save_tokenizer(tokenizer, path="LuminaLM_text_token.json"):
    tokenizer.save(path)

# Load a tokenizer
def load_tokenizer(path="LuminaLM_text_token.json"):
    return Tokenizer.from_file(path)

if __name__ == "__main__":
    additional_directory = r"C:\\Users\\ASUS\\Desktop\\LuminaLM\\Data"  # Update with your directory path
    tokenizer, trainer = create_tokenizer(vocab_size=50000)
    train_tokenizer(tokenizer, trainer, additional_directory=additional_directory)
    save_tokenizer(tokenizer)
