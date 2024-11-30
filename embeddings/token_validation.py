import logging
import torch
import traceback
from tokenizers import Tokenizer
from tokenizer import (
    TokenizationUtilities,
    HybridTokenizationStrategy,
    MedicalTokenizer
)

# Set up logging
LOG_FILE = "validation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def main():
    """Validation script for the medical tokenizer."""
    try:
        # Load the trained tokenizer
        tokenizer_path = "Medical_tokenizer.json"
        logging.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        medical_tokenizer = MedicalTokenizer()
        medical_tokenizer.tokenizer = tokenizer
        medical_tokenizer.strategy = HybridTokenizationStrategy(tokenizer)

        # Sample texts for validation
        sample_texts = [
            "Patient presents with severe chest pain.",
            "Medical history includes hypertension and diabetes.",
            "The patient was administered 5mg of medication Atorvastatin.",
            "Symptoms include fever, cough, and shortness of breath."
        ]

        # Perform autoregressive encoding
        logging.info("Performing autoregressive encoding...")
        auto_encoding = medical_tokenizer.encode(sample_texts, task_type='auto')
        logging.info(f"Autoregressive Encoding Output:\n{auto_encoding}")

        # Perform bidirectional encoding
        logging.info("Performing bidirectional encoding...")
        bi_encoding = medical_tokenizer.encode(sample_texts, task_type='bi')
        logging.info(f"Bidirectional Encoding Output:\n{bi_encoding}")

        # Generate masked language model inputs
        logging.info("Generating masked language model inputs...")
        utilities = TokenizationUtilities()
        special_token_ids = [medical_tokenizer.tokenizer.token_to_id(token) for token in medical_tokenizer.special_tokens]

        masked_inputs, labels, mask = utilities.generate_masked_lm_inputs(
            auto_encoding['input_ids'],
            mask_probability=0.15,
            mask_token_id=medical_tokenizer.tokenizer.token_to_id("[MASK]"),
            special_token_ids=special_token_ids,
            vocab_size=medical_tokenizer.vocab_size
        )

        logging.info(f"Masked Inputs:\n{masked_inputs}")
        logging.info(f"Labels:\n{labels}")
        logging.info(f"Mask:\n{mask}")

        # Validate that masked positions are consistent
        mask_positions = mask.nonzero(as_tuple=True)
        logging.info(f"Masked Positions: {mask_positions}")

        # Ensure that labels are -100 for unmasked tokens
        unmasked_labels = labels[~mask]
        if torch.all(unmasked_labels == -100):
            logging.info("Unmasked labels are correctly set to -100.")
        else:
            logging.warning("There are unmasked labels not set to -100.")

        logging.info("Validation completed successfully.")

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
