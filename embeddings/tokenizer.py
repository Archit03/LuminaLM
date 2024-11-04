import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from datasets import load_dataset
import os

# Create a function to handle tokenization
def create_tokens(vocab_size=50000, special_tokens=None):
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
    tokenizer.pre_tokenizer = ByteLevel()  # Better subword and punctuation handling
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    return tokenizer, trainer

# Function to preprocess OpenWebText data
def preprocess_openwebtext():
    dataset = load_dataset("openwebtext", split="train")
    texts = [item["text"] for item in dataset]  # Extract the text field for each item in the dataset
    return texts

# Train the tokenizer on OpenWebText and any additional local data
def train_tokenizer(tokenizer, trainer, additional_directory=None):
    # Load OpenWebText data
    openwebtext_data = preprocess_openwebtext()
    
    # Load and preprocess additional local data if directory is provided
    additional_texts = []
    if additional_directory:
        additional_texts = preprocess_files_from_directory(additional_directory)
    
    # Combine OpenWebText and local data
    combined_data = openwebtext_data + additional_texts

    # Train the tokenizer
    if combined_data:
        tokenizer.train_from_iterator(combined_data, trainer)
    else:
        raise ValueError("No valid text data found for training.")

# Preprocess files from a local directory
def preprocess_files_from_directory(directory):
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

# Custom function for preprocessing CSV files
def preprocess_csv(file):
    import pandas as pd
    df = pd.read_csv(file)
    text_data = " ".join(df.astype(str).values.flatten())
    return text_data

# Save the tokenizer to a file
def save_tokenizer(tokenizer, path="LuminaLM_text_token.json"):
    tokenizer.save(path)

# Load the tokenizer from a saved file
def load_tokenizer(path="LuminaLM_text_token.json"):
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    additional_directory = r'/home/ubuntu/LuminaLM/Data'  # Adjust to your actual directory path if needed
    tokenizer, trainer = create_tokens(vocab_size=50000)
    train_tokenizer(tokenizer, trainer, additional_directory=additional_directory)
    save_tokenizer(tokenizer)
