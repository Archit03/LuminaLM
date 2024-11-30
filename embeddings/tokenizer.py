import logging
import os
import json
import traceback
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch import Tensor
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Set up logging
LOG_FILE = "medical_tokenization.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

class TokenizationUtilities:
    """
    Tokenization utilities for complex NLP preprocessing.
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

        :param input_ids: List of input token IDs
        :param attention_masks: List of attention masks
        :param padding_value: Value to use for padding
        :param padding_side: Side to pad ('left' or 'right')
        :param max_length: Maximum sequence length
        :return: Padded input_ids and attention_masks
        """
        # Determine max length if not specified
        if max_length is None:
            max_length = max(len(ids) for ids in input_ids)

        # Pad input_ids
        padded_input_ids = []
        for ids in input_ids:
            pad_length = max_length - len(ids)
            if padding_side == 'right':
                padded_ids = torch.cat([
                    ids,
                    torch.full((pad_length,), padding_value, dtype=ids.dtype)
                ])
            else:  # left padding
                padded_ids = torch.cat([
                    torch.full((pad_length,), padding_value, dtype=ids.dtype),
                    ids
                ])
            padded_input_ids.append(padded_ids)

        # Pad attention masks
        padded_attention_masks = []
        for mask in attention_masks:
            pad_length = max_length - len(mask)
            if padding_side == 'right':
                padded_mask = torch.cat([
                    mask,
                    torch.zeros(pad_length, dtype=mask.dtype)
                ])
            else:  # left padding
                padded_mask = torch.cat([
                    torch.zeros(pad_length, dtype=mask.dtype),
                    mask
                ])
            padded_attention_masks.append(padded_mask)

        return torch.stack(padded_input_ids), torch.stack(padded_attention_masks)

    @staticmethod
    def create_segment_ids(
        input_ids: Tensor,
        separator_token_id: int,
        cls_token_id: int
    ) -> Tensor:
        """
        Create segment IDs for multi-segment inputs (e.g., sentence pairs).

        :param input_ids: Padded input token IDs
        :param separator_token_id: ID of the separator token
        :param cls_token_id: ID of the classification token
        :return: Tensor of segment IDs
        """
        segment_ids = torch.zeros_like(input_ids)
        for i, seq in enumerate(input_ids):
            # Find separator and CLS token positions
            sep_positions = (seq == separator_token_id).nonzero(as_tuple=True)[0]
            cls_position = (seq == cls_token_id).nonzero(as_tuple=True)[0]

            # Assign segment IDs
            if len(sep_positions) > 0:
                segment_ids[i, cls_position[0]:sep_positions[0] + 1] = 0
                segment_ids[i, sep_positions[0] + 1:] = 1

        return segment_ids

    @staticmethod
    def generate_masked_lm_inputs(
        input_ids: Tensor,
        mask_probability: float = 0.15,
        mask_token_id: int = 103,  # Typically the [MASK] token
        special_token_ids: Optional[List[int]] = None,
        vocab_size: int = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate masked language model inputs for pretraining.

        :param input_ids: Original input token IDs (batch_size x seq_length)
        :param mask_probability: Probability of masking a token
        :param mask_token_id: ID of the mask token
        :param special_token_ids: List of token IDs to never mask
        :param vocab_size: Vocabulary size for random token replacement
        :return: Masked input_ids, masked labels, and masking indicator
        """
        if special_token_ids is None:
            special_token_ids = []

        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        batch_size, seq_length = input_ids.shape

        # Create mask for tokens that can be masked
        maskable_positions = torch.ones_like(input_ids, dtype=torch.bool)
        for special_id in special_token_ids:
            maskable_positions &= (input_ids != special_id)

        # Randomly decide which tokens to mask
        probabilities = torch.full(input_ids.shape, mask_probability)
        mask_probabilities = torch.bernoulli(probabilities).bool() & maskable_positions

        # Apply masking
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & mask_probabilities
        masked_input_ids[indices_replaced] = mask_token_id

        # Apply random token replacement
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & mask_probabilities & ~indices_replaced
        if vocab_size is None:
            vocab_size = int(input_ids.max()) + 1
        random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, dtype=input_ids.dtype)
        masked_input_ids[indices_random] = random_tokens[indices_random]

        # The rest 10% keep the original tokens
        # For tokens not masked, set labels to -100 so they are ignored in loss calculation
        labels[~mask_probabilities] = -100

        return masked_input_ids, labels, mask_probabilities

class HybridTokenizationStrategy:
    """Tokenization strategy supporting both autoregressive and bidirectional processing."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.utilities = TokenizationUtilities()

    def autoregressive_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True,
        padding: bool = True
    ):
        """Encode text for left-to-right prediction tasks with causal masking."""
        encodings = self.tokenizer.encode_batch(texts)
        input_ids, attention_masks = [], []

        for encoding in encodings:
            ids = encoding.ids[:max_length] if truncation else encoding.ids
            input_ids.append(torch.tensor(ids))
            seq_len = len(ids)
            mask = torch.ones(seq_len, dtype=torch.long)
            attention_masks.append(mask)

        # Apply dynamic padding if required
        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(
                input_ids,
                attention_masks
            )

        # Create causal masks
        causal_masks = []
        for seq_len in attention_masks.sum(dim=1).tolist():
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            pad_size = input_ids.shape[1] - seq_len
            causal_mask = torch.nn.functional.pad(causal_mask, (0, pad_size, 0, pad_size), value=0)
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
    ):
        """Encode text with full contextual understanding for bidirectional tasks."""
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens
        )
        input_ids, attention_masks = [], []

        for encoding in encodings:
            ids = encoding.ids[:max_length] if truncation else encoding.ids
            input_ids.append(torch.tensor(ids))
            mask = torch.ones(len(ids), dtype=torch.long)
            attention_masks.append(mask)

        # Apply dynamic padding if required
        if padding:
            input_ids, attention_masks = self.utilities.dynamic_padding(
                input_ids,
                attention_masks
            )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }

    def hybrid_encode(
        self,
        texts: List[str],
        task_type: str = 'auto',
        **kwargs
    ):
        """Dynamically choose encoding strategy based on task type."""
        if task_type == 'auto':
            return self.autoregressive_encode(texts, **kwargs)
        elif task_type == 'bi':
            return self.bidirectional_encode(texts, **kwargs)
        else:
            raise ValueError(f"Invalid task type: {task_type}. Use 'auto' or 'bi'.")

class MedicalTokenizer:
    """Tokenizer class that combines both autoregressive and bidirectional encoding strategies."""

    def __init__(self, vocab_size: int = 60000, min_frequency: int = 2):
        self.tokenizer = Tokenizer(BPE())
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
        Train the tokenizer on the given files and save it to the specified path.

        :param files: List of file paths to use for training
        :param save_path: Path to save the trained tokenizer
        """
        logging.info("Starting tokenizer training...")
        self.tokenizer.train(files, self.trainer)
        self.tokenizer.save(save_path)
        logging.info(f"Tokenizer saved to {save_path}")

    def encode(self, texts: List[str], task_type: str = 'auto', **kwargs):
        """
        Encode a list of texts using the specified task type ('auto' or 'bi').

        :param texts: List of texts to encode
        :param task_type: 'auto' for autoregressive or 'bi' for bidirectional encoding
        :return: Encoded inputs
        """
        return self.strategy.hybrid_encode(texts, task_type=task_type, **kwargs)

class DatasetProcessor:
    """Class to process and prepare datasets for tokenizer training."""

    def __init__(self, datasets: List[Dict[str, Any]], local_data_path: str):
        self.datasets = datasets
        self.local_data_path = local_data_path
        self.processed_files = []

    def process(self) -> List[str]:
        """
        Process all datasets and return a list of file paths containing the processed data.

        :return: List of file paths
        """
        logging.info("Starting dataset processing...")
        os.makedirs(self.local_data_path, exist_ok=True)
        for dataset_info in self.datasets:
            dataset_name = dataset_info.get('name')
            configs = dataset_info.get('configs', [None])
            trust_remote_code = dataset_info.get('trust_remote_code', False)
            for config in configs:
                logging.info(f"Processing dataset: {dataset_name}, config: {config}")
                try:
                    dataset = load_dataset(
                        dataset_name,
                        config,
                        split='train',
                        cache_dir=self.local_data_path,
                        download_mode='reuse_dataset_if_exists',
                        trust_remote_code=trust_remote_code
                        # Removed ignore_verifications=True
                    )
                except TypeError as te:
                    logging.warning(f"'ignore_verifications' parameter is not supported for dataset {dataset_name} with config {config}. Proceeding without it.")
                    dataset = load_dataset(
                        dataset_name,
                        config,
                        split='train',
                        cache_dir=self.local_data_path,
                        download_mode='reuse_dataset_if_exists',
                        trust_remote_code=trust_remote_code
                    )
                file_path = self.save_dataset_to_file(dataset, dataset_name, config)
                self.processed_files.append(file_path)
        logging.info("Dataset processing completed.")
        return self.processed_files

    def save_dataset_to_file(self, dataset, dataset_name, config) -> str:
        """
        Save the dataset to a text file for tokenizer training.

        :param dataset: The dataset to save
        :param dataset_name: Name of the dataset
        :param config: Configuration of the dataset
        :return: Path to the saved file
        """
        file_name = f"{dataset_name}_{config}_train.txt" if config else f"{dataset_name}_train.txt"
        file_path = os.path.join(self.local_data_path, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                text = self.extract_text(item)
                if text:
                    f.write(text + '\n')
        logging.info(f"Saved dataset to {file_path}")
        return file_path

    @staticmethod
    def extract_text(item) -> str:
        """
        Extract text data from a dataset item.

        :param item: A single data item from the dataset
        :return: Extracted text
        """
        if 'text' in item:
            return item['text']
        elif 'question' in item and 'context' in item:
            return item['question'] + ' ' + item['context']
        elif 'abstract' in item:
            return item['abstract']
        else:
            return ''

def main():
    """Main execution for medical tokenizer training."""
    datasets = [
        {"name": "openwebtext", "trust_remote_code": True},
        {"name": "pubmed_qa", "configs": ["pqa_artificial", "pqa_labeled", "pqa_unlabeled"]},
    ]
    local_data_path = "data/medical"

    dataset_processor = DatasetProcessor(datasets, local_data_path)
    tokenizer = MedicalTokenizer(vocab_size=60000, min_frequency=3)

    try:
        processed_data = dataset_processor.process()
        tokenizer.train(processed_data, "Medical_tokenizer.json")

        logging.info("Tokenizer training completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
