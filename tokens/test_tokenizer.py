import unittest
import tempfile
import torch
import os
from pathlib import Path
from unittest.mock import Mock, patch
from tokenizer import (
    TokenizationUtilities, 
    HybridTokenizationStrategy,
    MedicalTokenizer,
    TokenizerPathManager,
    DatasetProcessor,
    Config
)

class TestTokenizationUtilities(unittest.TestCase):
    def setUp(self):
        self.utils = TokenizationUtilities()
        self.device = torch.device('cpu')

    def test_dynamic_padding(self):
        # Test normal case
        input_ids = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2])
        ]
        attention_masks = [
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1])
        ]
        padded_ids, padded_masks = self.utils.dynamic_padding(input_ids, attention_masks)
        self.assertEqual(padded_ids.shape, (2, 3))
        self.assertEqual(padded_masks.shape, (2, 3))

        # Test empty input
        with self.assertRaises(ValueError):
            self.utils.dynamic_padding([], [])

        # Test mismatched lengths
        with self.assertRaises(ValueError):
            self.utils.dynamic_padding(
                [torch.tensor([1, 2])],
                [torch.tensor([1])]
            )

    def test_create_segment_ids(self):
        input_ids = torch.tensor([[101, 1, 2, 102, 3, 4, 102]])
        segment_ids = self.utils.create_segment_ids(
            input_ids=input_ids,
            separator_token_id=102,
            cls_token_id=101
        )
        expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
        torch.testing.assert_close(segment_ids, expected)

    def test_generate_masked_lm_inputs(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        masked_ids, labels, mask_probs = self.utils.generate_masked_lm_inputs(
            input_ids=input_ids,
            mask_probability=1.0,  # Force masking for testing
            mask_token_id=103,
            special_token_ids=[1, 5]  # Don't mask first and last tokens
        )
        # Check that first and last tokens weren't masked
        self.assertEqual(masked_ids[0, 0].item(), 1)
        self.assertEqual(masked_ids[0, -1].item(), 5)


class TestHybridTokenizationStrategy(unittest.TestCase):
    @patch('tokenizer.Tokenizer')
    def setUp(self, MockTokenizer):
        self.mock_tokenizer = MockTokenizer()
        self.strategy = HybridTokenizationStrategy(self.mock_tokenizer)

    def test_autoregressive_encode(self):
        self.mock_tokenizer.encode_batch.return_value = [
            Mock(ids=[1, 2, 3]),
            Mock(ids=[1, 2])
        ]
        result = self.strategy.autoregressive_encode(
            texts=["test1", "test2"],
            max_length=5
        )
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('causal_mask', result)

    def test_bidirectional_encode(self):
        self.mock_tokenizer.encode_batch.return_value = [
            Mock(ids=[1, 2, 3]),
            Mock(ids=[1, 2])
        ]
        result = self.strategy.bidirectional_encode(
            texts=["test1", "test2"],
            max_length=5
        )
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)


class TestTokenizerPathManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path_manager = TokenizerPathManager(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_validate_save_path(self):
        # Test valid path
        save_path = Path(self.temp_dir) / "tokenizer.json"
        validated_path = self.path_manager.validate_save_path(save_path)
        self.assertEqual(validated_path, save_path)

        # Test read-only directory
        if os.name != 'nt':  # Skip on Windows
            read_only_dir = Path(self.temp_dir) / "readonly"
            read_only_dir.mkdir()
            os.chmod(read_only_dir, 0o444)
            with self.assertRaises(PermissionError):
                self.path_manager.validate_save_path(read_only_dir / "tokenizer.json")

    def test_get_backup_path(self):
        original_path = Path(self.temp_dir) / "tokenizer.json"
        backup_path = self.path_manager.get_backup_path(original_path)
        self.assertIn("backup", backup_path.name)
        self.assertTrue(backup_path.name.endswith(".json"))


class TestDatasetProcessor(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            local_data_path=tempfile.mkdtemp(),
            vocab_size=1000,
            min_frequency=2
        )
        self.datasets = [
            {"path": "test.txt"}
        ]
        self.processor = DatasetProcessor(self.datasets, self.config)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.config.local_data_path)

    def test_extract_text(self):
        # Test string input
        text = self.processor.extract_text("Simple text")
        self.assertEqual(text, "Simple text")

        # Test dict input
        record = {
            "text": "Main text",
            "description": "Description"
        }
        text = self.processor.extract_text(record)
        self.assertTrue("Main text" in text)
        self.assertTrue("Description" in text)

    def test_preprocess_text(self):
        text = "Test with URL http://example.com and email test@example.com"
        processed = self.processor.preprocess_text(text)
        self.assertNotIn("http://", processed)
        self.assertNotIn("@example.com", processed)

    @patch('tokenizer.ProcessPoolExecutor')
    def test_parallel_process_texts(self, MockExecutor):
        mock_executor = MockExecutor.return_value.__enter__.return_value
        mock_executor.submit.return_value.result.return_value = ["processed"]
        
        texts = ["test1", "test2"]
        result = self.processor._parallel_process_texts(texts, chunk_size=1)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main() 