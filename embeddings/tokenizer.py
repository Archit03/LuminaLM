import os
import re
import logging
import json
from typing import List, Dict, Any, Callable, Iterator, Optional
from datetime import datetime

import spacy
import unidecode
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset

class MedicalTextPreprocessor:
    """Comprehensive preprocessor for medical text with configurable rules."""
    
    def __init__(self, 
                 custom_rules: Optional[List[Dict[str, Any]]] = None,
                 min_token_length: int = 2,
                 max_token_length: int = 30,
                 use_medical_nlp: bool = True):
        """
        Initialize the medical text preprocessor.
        
        Args:
            custom_rules: List of custom regex replacement rules
            min_token_length: Minimum length of tokens to keep
            max_token_length: Maximum length of tokens to keep
            use_medical_nlp: Whether to use medical NLP for normalization
        """
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Default medical-specific normalization rules
        self.default_rules = [
            # Medical units and measurements
            {"pattern": r'(\d+)\s*°\s*C', "repl": r'\1°C'},
            {"pattern": r'(\d+)\s*mmHg', "repl": r'\1mmHg'},
            {"pattern": r'(\d+)\s*kg/m²', "repl": r'\1kg/m²'},
            {"pattern": r'(\d+)\s*bpm', "repl": r'\1bpm'},
            
            # Medical abbreviations
            {"pattern": r'\bb\.?i\.?d\b', "repl": 'twice daily', "flags": re.IGNORECASE},
            {"pattern": r'\bt\.?i\.?d\b', "repl": 'three times daily', "flags": re.IGNORECASE},
            {"pattern": r'\bq\.?d\b', "repl": 'daily', "flags": re.IGNORECASE},
            {"pattern": r'\bp\.?r\.?n\b', "repl": 'as needed', "flags": re.IGNORECASE},
            {"pattern": r'\bp\.?o\b', "repl": 'by mouth', "flags": re.IGNORECASE},
        ]
        
        # Merge custom rules with default rules
        self.rules = custom_rules or []
        self.rules.extend(self.default_rules)
        
        # Initialize medical NLP if requested
        self.nlp = None
        if use_medical_nlp:
            try:
                self.nlp = spacy.load('en_core_sci_md')
            except OSError:
                logging.warning("Medical NLP model not found. Falling back to standard preprocessing.")
    
    def _apply_regex_rules(self, text: str) -> str:
        """Apply regex replacement rules."""
        for rule in self.rules:
            text = re.sub(
                rule['pattern'], 
                rule['repl'], 
                text, 
                flags=rule.get('flags', 0)
            )
        return text
    
    def _medical_nlp_normalize(self, text: str) -> str:
        """Normalization using medical NLP."""
        if not self.nlp:
            return text
        
        doc = self.nlp(text)
        normalized_tokens = []
        
        for token in doc:
            # Lemmatization
            lemma = token.lemma_
            
            # Normalization techniques
            normalized = unidecode.unidecode(lemma.lower())
            
            # Length and content filtering
            if (self.min_token_length <= len(normalized) <= self.max_token_length 
                and not token.is_punct 
                and not token.is_space):
                normalized_tokens.append(normalized)
        
        return ' '.join(normalized_tokens)
    
    def preprocess(self, text: str) -> str:
        """
        Comprehensive text preprocessing method.
        
        Args:
            text: Input text to preprocess
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Apply regex rules
        text = self._apply_regex_rules(text)
        
        # Apply medical NLP normalization if available
        text = self._medical_nlp_normalize(text)
        
        return text


class MedicalTokenizer:
    """Robust medical tokenizer with comprehensive configuration options."""
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 min_frequency: int = 2,
                 custom_preprocessing_rules: Optional[List[Dict[str, Any]]] = None,
                 local_data_path: str = "",
                 preprocessor_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the medical tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary
            min_frequency: Minimum token frequency to include in vocab
            custom_preprocessing_rules: Custom regex preprocessing rules
            local_data_path: Path to local data files
            preprocessor_kwargs: Additional kwargs for preprocessor
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.local_data_path = local_data_path
        
        # Initialize preprocessor
        preprocessor_kwargs = preprocessor_kwargs or {}
        if custom_preprocessing_rules:
            preprocessor_kwargs['custom_rules'] = custom_preprocessing_rules
        
        self.preprocessor = MedicalTextPreprocessor(**preprocessor_kwargs)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    
    def _configure_tokenizer(self):
        """Configure tokenizer settings."""
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")), 
                ("</s>", self.tokenizer.token_to_id("</s>"))
            ]
        )
    
    def _stream_datasets(self, datasets):
        """Stream and preprocess datasets with progress tracking."""
        for dataset_info in tqdm(datasets, desc="Processing Datasets"):
            try:
                ds = load_dataset(dataset_info['name'], split='train')
                for example in tqdm(ds, desc=f"Processing {dataset_info['name']}", leave=False):
                    text = example.get('text', '') or example.get('content', '')
                    if text:
                        yield self.preprocessor.preprocess(text)
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_info['name']}: {e}")
    
    def train(self, 
              datasets: List[Dict[str, Any]], 
              output_path: Optional[str] = None,
              max_training_steps: Optional[int] = None):
        """
        Train the tokenizer with comprehensive configuration.
        
        Args:
            datasets: List of dataset configurations
            output_path: Path to save the trained tokenizer
            max_training_steps: Optional maximum training steps
        """
        self._configure_tokenizer()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=self.special_tokens,
            min_frequency=self.min_frequency
        )
        
        logging.info("Starting tokenizer training...")
        self.tokenizer.train_from_iterator(
            self._stream_datasets(datasets), 
            trainer=trainer
        )
        
        logging.info("Tokenizer training complete.")
        
        if output_path:
            self.save(output_path)
    
    def save(self, path: str):
        """Save the trained tokenizer."""
        self.tokenizer.save(path)
        logging.info(f"Tokenizer saved to {path}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Custom preprocessing rules example
    custom_rules = [
        {"pattern": r'\bBP\b', "repl": 'blood pressure', "flags": re.IGNORECASE},
        {"pattern": r'\bhr\b', "repl": 'heart rate', "flags": re.IGNORECASE}
    ]
    
    # Datasets
    datasets = [
        {"name": "rungalileo/medical_transcription_40"},
        {"name": "gamino/wiki_medical_terms"},
        {"name": "medalpaca/medical_meadow_medqa"},
        {"name": "openwebtext"}
    ]
    
    # Initialize tokenizer with comprehensive configuration
    tokenizer = MedicalTokenizer(
        vocab_size=60000,
        min_frequency=3,
        custom_preprocessing_rules=custom_rules,
        preprocessor_kwargs={
            "min_token_length": 2,
            "max_token_length": 25,
            "use_medical_nlp": True
        }
    )
    
    # Train and save
    tokenizer.train(
        datasets, 
        output_path="medical_tokenizer.json"
    )


if __name__ == "__main__":
    main()