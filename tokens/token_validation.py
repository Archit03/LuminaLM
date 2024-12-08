import logging
import torch
import traceback
import os
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
        
        # Add file existence check
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        tokenizer = Tokenizer.from_file(tokenizer_path)
        medical_tokenizer = MedicalTokenizer()
        medical_tokenizer.tokenizer = tokenizer
        medical_tokenizer.strategy = HybridTokenizationStrategy(tokenizer)

        # Expanded sample texts with more diverse medical content
        sample_texts = [
            "Patient presents with severe chest pain.",
            "Medical history includes hypertension and diabetes.",
            "The patient was administered 5mg of medication Atorvastatin.",
            "Symptoms include fever, cough, and shortness of breath.",
            "Lab results: WBC 12.3, RBC 4.5, Platelets 250k",
            "Patient reports allergies to penicillin and sulfa drugs",
            "Post-operative recovery shows good wound healing"
        ]

        # Add validation metrics
        def validate_encoding(encoding, task_type):
            logging.info(f"\nValidating {task_type} encoding:")
            logging.info(f"Shape of input_ids: {encoding['input_ids'].shape}")
            logging.info(f"Sequence lengths: {encoding['input_ids'].sum(dim=1)}")
            logging.info(f"Attention mask statistics: {encoding['attention_mask'].float().mean():.2f} coverage")
            
            # Validate special tokens
            special_token_count = sum(1 for id in encoding['input_ids'].flatten() 
                                    if id in special_token_ids)
            logging.info(f"Special tokens found: {special_token_count}")

        # Perform and validate autoregressive encoding
        logging.info("\nPerforming autoregressive encoding...")
        auto_encoding = medical_tokenizer.encode(sample_texts, task_type='auto')
        validate_encoding(auto_encoding, 'autoregressive')

        # Perform and validate bidirectional encoding
        logging.info("\nPerforming bidirectional encoding...")
        bi_encoding = medical_tokenizer.encode(sample_texts, task_type='bi')
        validate_encoding(bi_encoding, 'bidirectional')

        # Enhanced MLM validation
        logging.info("\nGenerating masked language model inputs...")
        utilities = TokenizationUtilities()
        special_token_ids = [medical_tokenizer.tokenizer.token_to_id(token) 
                            for token in medical_tokenizer.special_tokens]

        masked_inputs, labels, mask = utilities.generate_masked_lm_inputs(
            auto_encoding['input_ids'],
            mask_probability=0.15,
            mask_token_id=medical_tokenizer.tokenizer.token_to_id("[MASK]"),
            special_token_ids=special_token_ids,
            vocab_size=medical_tokenizer.vocab_size
        )

        # Add detailed MLM statistics
        total_tokens = mask.numel()
        masked_tokens = mask.sum().item()
        mask_percentage = (masked_tokens / total_tokens) * 100

        logging.info(f"\nMLM Statistics:")
        logging.info(f"Total tokens: {total_tokens}")
        logging.info(f"Masked tokens: {masked_tokens}")
        logging.info(f"Actual mask percentage: {mask_percentage:.2f}%")
        
        # Validate mask consistency
        mask_positions = mask.nonzero(as_tuple=True)
        label_consistency = torch.all(labels[mask] != -100).item()
        logging.info(f"Labels at masked positions are valid: {label_consistency}")

        logging.info("Validation completed successfully.")

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
