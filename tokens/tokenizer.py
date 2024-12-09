import argparse
import logging
import os
import json
import mimetypes
import hashlib
import multiprocessing
import traceback
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm
import torch
from torch import Tensor
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import pandas as pd
from PIL import Image
import io
from transformers import ViTImageProcessor, CLIPProcessor

###############################################################################
# Configuration
###############################################################################
class Config:
    """
    Configuration class to hold parameters for dataset processing and tokenizer training.
    """
    def __init__(
        self, 
        local_data_path: str,
        vocab_size: int = 60000,
        min_frequency: int = 2,
        max_file_size_mb: int = 100,
        allowed_extensions: Set[str] = {'.txt', '.csv', '.json', '.jsonl', '.text', 
                                      '.jpg', '.jpeg', '.png', '.webp'},
        allowed_mimetypes: Set[str] = {
            'text/plain', 'text/csv', 'application/json',
            'application/x-json', 'text/json',
            'image/jpeg', 'image/png', 'image/webp'
        },
        image_processor: str = 'google/vit-base-patch16-224',
        chunk_size: int = 1000,
        processing_workers: int = max(1, multiprocessing.cpu_count() - 1),
        log_file: str = "medical_tokenization.log"
    ):
        self.local_data_path = Path(local_data_path)
        self.local_data_path.mkdir(parents=True, exist_ok=True)
        (self.local_data_path / "tokens").mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.allowed_extensions = allowed_extensions
        self.allowed_mimetypes = allowed_mimetypes
        self.chunk_size = chunk_size
        self.processing_workers = processing_workers
        self.log_file = log_file
        self.image_processor = image_processor


###############################################################################
# Logging Setup
###############################################################################
def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

###############################################################################
# Tokenization Utilities
###############################################################################
class TokenizationUtilities:
    """
    Utilities for NLP preprocessing: padding, segment IDs, and masked LM inputs.
    """

    @staticmethod
    def dynamic_padding(
        input_ids: List[Tensor],
        attention_masks: List[Tensor],
        padding_value: int = 0,
        padding_side: str = 'right',
        max_length: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Dynamically pad input sequences to a consistent length.
        """
        if not input_ids:
            raise ValueError("No input_ids provided for padding.")
        
        device = input_ids[0].device
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(input_ids, attention_masks):
            pad_length = max_length - len(ids)
            if padding_side == 'right':
                padded_ids = torch.cat([ids, torch.full((pad_length,), padding_value, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)])
            else:
                padded_ids = torch.cat([torch.full((pad_length,), padding_value, dtype=ids.dtype), ids])
                padded_mask = torch.cat([torch.zeros(pad_length, dtype=mask.dtype), mask])
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        return (torch.stack(padded_input_ids).to(device), torch.stack(padded_attention_masks).to(device))

    @staticmethod
    def create_segment_ids(
        input_ids: Tensor,
        separator_token_id: int,
        cls_token_id: int
    ) -> Tensor:
        """
        Create segment IDs for multi-segment inputs (e.g., sentence pairs).
        """
        device = input_ids.device
        segment_ids = torch.zeros_like(input_ids).to(device)
        for i, seq in enumerate(input_ids):
            sep_positions = (seq == separator_token_id).nonzero(as_tuple=True)[0]
            cls_positions = (seq == cls_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0 and len(cls_positions) > 0:
                # For simplicity, assume only one CLS at start
                segment_ids[i, cls_positions[0]:sep_positions[0] + 1] = 0
                if sep_positions[0] + 1 < len(seq):
                    segment_ids[i, sep_positions[0] + 1:] = 1
        return segment_ids

    @staticmethod
    def generate_masked_lm_inputs(
        input_ids: Tensor,
        mask_probability: float = 0.15,
        mask_token_id: int = 103,
        special_token_ids: Optional[List[int]] = None,
        vocab_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate masked LM inputs using a standard BERT-style masking strategy:
        80% [MASK], 10% random, 10% original token.
        """
        if special_token_ids is None:
            special_token_ids = []
        
        device = input_ids.device
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        batch_size, seq_length = input_ids.shape

        # Identify maskable positions
        maskable_positions = torch.ones_like(input_ids, dtype=torch.bool)
        for sid in special_token_ids:
            maskable_positions &= (input_ids != sid)

        probabilities = torch.full(input_ids.shape, mask_probability, device=device)
        mask_probabilities = (torch.bernoulli(probabilities).bool()) & maskable_positions

        # 80% of the masked positions -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8, device=device)).bool() & mask_probabilities
        masked_input_ids[indices_replaced] = mask_token_id

        # 10% of the masked positions -> random tokens
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5, device=device)).bool() & mask_probabilities & ~indices_replaced
        if vocab_size is None:
            vocab_size = int(input_ids.max()) + 1
        random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=input_ids.dtype, device=device)
        masked_input_ids[indices_random] = random_tokens[indices_random]

        # The remaining 10% stay the same. 
        # For non-masked positions, set labels to -100 so they are ignored in loss
        labels[~mask_probabilities] = -100

        return masked_input_ids, labels, mask_probabilities


###############################################################################
# Hybrid Tokenization Strategy
###############################################################################
class HybridTokenizationStrategy:
    """
    Tokenization strategy supporting both autoregressive and bidirectional processing.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.utilities = TokenizationUtilities()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def autoregressive_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, Tensor]:
        """
        Encode text for autoregressive (causal) tasks.
        """
        encodings = self.tokenizer.encode_batch(texts)
        input_ids_list, attention_masks = [], []
        for encoding in encodings:
            ids = encoding.ids
            if truncation:
                ids = ids[:max_length]
            tensor_ids = torch.tensor(ids, device=self.device)
            mask = torch.ones(len(tensor_ids), dtype=torch.long, device=self.device)
            input_ids_list.append(tensor_ids)
            attention_masks.append(mask)

        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(input_ids_list, attention_masks)
        else:
            input_ids = torch.stack(input_ids_list)
            attention_masks = torch.stack(attention_masks)

        # Create causal masks
        causal_masks = []
        seq_lengths = attention_masks.sum(dim=1).tolist()
        for seq_len in seq_lengths:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float, device=self.device))
            pad_len = input_ids.shape[1] - seq_len
            # Pad the causal mask to fit the full sequence length
            causal_mask = torch.nn.functional.pad(causal_mask, (0, pad_len, 0, pad_len), value=0)
            causal_masks.append(causal_mask)

        causal_masks = torch.stack(causal_masks)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'causal_mask': causal_masks
        }

    def bidirectional_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = True
    ) -> Dict[str, Tensor]:
        """
        Encode text for bidirectional tasks.
        """
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        input_ids_list, attention_masks = [], []
        for encoding in encodings:
            ids = encoding.ids
            if truncation:
                ids = ids[:max_length]
            tensor_ids = torch.tensor(ids, device=self.device)
            mask = torch.ones(len(tensor_ids), dtype=torch.long, device=self.device)
            input_ids_list.append(tensor_ids)
            attention_masks.append(mask)

        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(input_ids_list, attention_masks)
        else:
            input_ids = torch.stack(input_ids_list)
            attention_masks = torch.stack(attention_masks)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def hybrid_encode(
        self,
        texts: List[str],
        task_type: str = 'auto',
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        Dynamically choose encoding strategy based on task type ('auto' or 'bi').
        """
        if task_type == 'auto':
            return self.autoregressive_encode(texts, **kwargs)
        elif task_type == 'bi':
            return self.bidirectional_encode(texts, **kwargs)
        else:
            raise ValueError(f"Invalid task type: {task_type}. Use 'auto' or 'bi'.")


###############################################################################
# MedicalTokenizer
###############################################################################
class MedicalTokenizer:
    """
    Tokenizer class for training and encoding with both autoregressive and bidirectional strategies.
    """

    def __init__(self, vocab_size: int = 60000, min_frequency: int = 2):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        self.strategy = HybridTokenizationStrategy(self.tokenizer)

    def train(self, files: List[str], save_path: str):
        """
        Train the tokenizer on the given files and save it.
        """
        save_path = Path(save_path)
        try:
            logging.info("Starting tokenizer training...")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.tokenizer.train(files, self.trainer)
            self.tokenizer.save(str(save_path))
            logging.info(f"Tokenizer saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to train/save tokenizer: {e}")
            raise

    def load(self, tokenizer_path: str):
        """
        Load a previously trained tokenizer.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.strategy = HybridTokenizationStrategy(self.tokenizer)

    def encode(self, texts: List[str], task_type: str = 'auto', **kwargs) -> Dict[str, Tensor]:
        """
        Encode texts using the given task type ('auto' or 'bi').
        """
        return self.strategy.hybrid_encode(texts, task_type=task_type, **kwargs)


###############################################################################
# File Validator
###############################################################################
class FileValidator:
    """
    Validates and sanitizes file uploads.
    """
    def __init__(self, allowed_extensions: Set[str], allowed_mimetypes: Set[str], max_file_size: int):
        self.allowed_extensions = allowed_extensions
        self.allowed_mimetypes = allowed_mimetypes
        self.max_file_size = max_file_size

    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                return False, "File does not exist"
            if not file_path.is_file():
                return False, "Path is not a file"
            if file_path.suffix.lower() not in self.allowed_extensions:
                return False, f"Unsupported file extension. Allowed: {self.allowed_extensions}"
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type not in self.allowed_mimetypes:
                return False, f"Invalid file type: {mime_type}"
            if file_path.stat().st_size > self.max_file_size:
                return False, f"File too large. Maximum size: {self.max_file_size / 1024 / 1024}MB"
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        ext = Path(filename).suffix
        safe_name = f"file_{filename_hash}{ext}"
        return safe_name


###############################################################################
# Dataset Processor
###############################################################################
class DatasetProcessor:
    """
    Processes datasets/files into text for tokenizer training, with parallelization and error handling.
    """
    def __init__(self, 
                 datasets: List[Dict[str, Any]], 
                 config: Config):
        self.datasets = datasets
        self.config = config
        self.file_validator = FileValidator(
            allowed_extensions=config.allowed_extensions,
            allowed_mimetypes=config.allowed_mimetypes,
            max_file_size=config.max_file_size
        )
        self.config.local_data_path.mkdir(parents=True, exist_ok=True)
        self.image_processor = ViTImageProcessor.from_pretrained(config.image_processor)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process(self) -> List[str]:
        """
        Process all datasets and return a list of processed file paths.
        For Hugging Face datasets, load and process them.
        For local files, just process them directly.
        """
        processed_files = []

        for dataset_info in self.datasets:
            # Determine if it's a Hugging Face dataset or a local file
            if "name" in dataset_info and not dataset_info.get("path"):
                # Hugging Face dataset
                processed_files.extend(self._process_hf_dataset(dataset_info))
            elif "path" in dataset_info:
                # Local file
                file_path = dataset_info["path"]
                processed = self._process_local_file(file_path)
                if processed is not None:
                    processed_files.append(processed)
            else:
                logging.warning(f"Dataset info not recognized: {dataset_info}")

        return processed_files

    def _process_hf_dataset(self, dataset_info: Dict[str, Any]) -> List[str]:
        """
        Process a Hugging Face dataset (download and iterate over splits).
        """
        name = dataset_info["name"]
        configs = dataset_info.get("configs", [])
        processed_files = []
        
        try:
            if configs:
                for config_name in configs:
                    ds = load_dataset(name, config_name)
                    for split in ds.keys():
                        data = ds[split]
                        processed = self._process_hf_split(data, f"{name}_{config_name}_{split}")
                        if processed:
                            processed_files.append(processed)
            else:
                ds = load_dataset(name)
                for split in ds.keys():
                    data = ds[split]
                    processed = self._process_hf_split(data, f"{name}_{split}")
                    if processed:
                        processed_files.append(processed)

        except Exception as e:
            logging.error(f"Error processing dataset {name}: {str(e)}")
            return []

        return processed_files

    def _process_hf_split(self, hf_dataset, prefix: str) -> Optional[str]:
        """
        Process a single split of a Hugging Face dataset.
        """
        output_file = self.config.local_data_path / f"processed_{prefix}.txt"
        try:
            records = [self.extract_text(x) for x in hf_dataset]
            records = [r for r in records if r.strip()]

            # Process in parallel chunks
            processed_texts = self._parallel_process_texts(records, self.config.chunk_size)

            with open(output_file, 'w', encoding='utf-8') as f:
                for text in processed_texts:
                    f.write(text + "\n")

            return str(output_file)
        except Exception as e:
            logging.error(f"Error processing HF dataset {prefix}: {e}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _process_local_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Process a single local file (CSV, JSONL, TXT, or image) and return output path.
        """
        file_path = Path(file_path)
        is_valid, error_msg = self.file_validator.validate_file(file_path)
        if not is_valid:
            logging.error(f"File validation failed for {file_path}: {error_msg}")
            return None

        safe_filename = self.file_validator.sanitize_filename(file_path.name)
        output_file = self.config.local_data_path / f"processed_{safe_filename}"

        try:
            ext = file_path.suffix.lower()
            # Add image processing logic
            if ext in {'.jpg', '.jpeg', '.png', '.webp'}:
                return self._process_image_file(file_path, output_file.with_suffix('.json'))
            elif ext == '.csv':
                return self._process_csv_file(file_path, output_file)
            elif ext == '.jsonl':
                return self._process_jsonl_file(file_path, output_file)
            elif ext in {'.txt', '.text'}:
                return self._process_text_file(file_path, output_file)
            elif ext == '.json':
                # For .json, assume it's a list of objects
                return self._process_json_file(file_path, output_file)
            else:
                logging.warning(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            if output_file.exists():
                output_file.unlink()
            return None

    def _process_csv_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Process CSV files in chunks.
        """
        try:
            chunk_size = 10000
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in pd.read_csv(input_path, chunksize=chunk_size):
                    records = chunk.to_dict('records')
                    texts = [self.extract_text(r) for r in records if self.extract_text(r).strip()]
                    processed_texts = self._parallel_process_texts(texts, self.config.chunk_size)
                    for text in processed_texts:
                        f.write(text + "\n")
            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing CSV file {input_path}: {str(e)}")
            return None

    def _process_jsonl_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Process JSONL files line-by-line.
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                lines = []
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = self.extract_text(obj)
                        if text.strip():
                            lines.append(text)
                    except json.JSONDecodeError:
                        continue

            # Process in parallel
            processed_texts = self._parallel_process_texts(lines, self.config.chunk_size)

            with open(output_path, 'w', encoding='utf-8') as f_out:
                for text in processed_texts:
                    f_out.write(text + "\n")

            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing JSONL file {input_path}: {str(e)}")
            return None

    def _process_json_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Process a .json file. Assuming it's a JSON array of objects.
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in:
                data = json.load(f_in)
                # data should be a list of objects
                if not isinstance(data, list):
                    logging.error(f"JSON file {input_path} does not contain a list of objects.")
                    return None
                texts = [self.extract_text(item) for item in data if self.extract_text(item).strip()]

            processed_texts = self._parallel_process_texts(texts, self.config.chunk_size)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for text in processed_texts:
                    f_out.write(text + "\n")

            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing JSON file {input_path}: {str(e)}")
            return None

    def _process_text_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Process text files line-by-line.
        """
        try:
            chunk_size = 10000
            lines = []
            with open(input_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    text = line.strip()
                    if text:
                        lines.append(text)

            # Process in parallel
            processed_texts = self._parallel_process_texts(lines, self.config.chunk_size)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for text in processed_texts:
                    f_out.write(text + "\n")
            return str(output_path)
        except Exception as e:
            logging.error(f"Error processing text file {input_path}: {str(e)}")
            return None

    def _process_image_file(self, input_path: Path, output_path: Path) -> Optional[str]:
        """
        Process image files by extracting features and converting to text representation.
        """
        try:
            # Load and process image
            image = Image.open(input_path).convert('RGB')
            
            # Get ViT features
            vit_inputs = self.image_processor(image, return_tensors="pt")
            vit_features = vit_inputs['pixel_values'].squeeze(0)
            
            # Get CLIP text features for zero-shot classification
            clip_inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            clip_features = clip_inputs['pixel_values'].squeeze(0)
            
            # Combine features and convert to string representation
            combined_features = {
                'vit_features': vit_features.tolist(),
                'clip_features': clip_features.tolist(),
                'image_size': image.size,
                'mode': image.mode
            }
            
            # Save features as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_features, f)
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Error processing image file {input_path}: {str(e)}")
            return None

    def _parallel_process_texts(self, texts: List[str], chunk_size: int) -> List[str]:
        """
        Process a list of texts in parallel chunks.
        """
        chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        processed_texts = []
        with ProcessPoolExecutor(max_workers=self.config.processing_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    try:
                        processed_texts.extend(future.result())
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"Error processing chunk: {str(e)}")
                        traceback.print_exc()
        return processed_texts

    @staticmethod
    def _process_chunk(chunk: List[str]) -> List[str]:
        """
        Process a chunk of text lines. 
        In this simplified example, 'processing' might be identity or basic cleaning.
        """
        # Here you can add complex NLP processing if needed.
        return [text.strip() for text in chunk if text.strip()]

    @staticmethod
    def extract_text(record: Any) -> str:
        """
        Extract text from a given record.
        This method needs to be customized depending on the dataset structure.
        For demonstration, we attempt to handle:
         - Strings directly
         - Dicts with a 'text' field
         - If not found, returns empty string
        """
        if isinstance(record, str):
            return record
        elif isinstance(record, dict):
            # Try the 'text' key
            return record.get('text', '')
        else:
            return ''


###############################################################################
# Main Execution
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train a medical tokenizer on given datasets.")
    parser.add_argument('--local_data_path', type=str, required=True, help="Path to store processed data and tokenizer.")
    parser.add_argument('--vocab_size', type=int, default=60000, help="Vocabulary size for the tokenizer.")
    parser.add_argument('--min_frequency', type=int, default=3, help="Minimum frequency for BPE merges.")
    parser.add_argument('--log_file', type=str, default="medical_tokenization.log", help="Log file name.")
    args = parser.parse_args()

    setup_logging(args.log_file)

    # Modified dataset configuration
    datasets = [
        {
            "name": "openwebtext",
            "configs": None,  # No specific config needed
        },
        {
            "name": "pubmed_qa",
            "configs": ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]
        },
        # Add more datasets as needed
    ]

    try:
        # Create necessary directories
        tokens_dir = Path(args.local_data_path) / "tokens"
        tokens_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        config = Config(
            local_data_path=args.local_data_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            log_file=args.log_file
        )

        dataset_processor = DatasetProcessor(datasets, config)
        tokenizer = MedicalTokenizer(vocab_size=config.vocab_size, min_frequency=config.min_frequency)

        # Process and train
        logging.info("Starting dataset processing...")
        processed_files = dataset_processor.process()
        if not processed_files:
            raise ValueError("No files were successfully processed")
        logging.info(f"Dataset processing completed. Processed {len(processed_files)} files.")
        
        tokenizer_path = tokens_dir / "Medical_tokenizer.json"
        tokenizer.train(processed_files, str(tokenizer_path))
        logging.info("Tokenizer training completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
