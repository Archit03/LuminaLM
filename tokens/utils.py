import tempfile
from pathlib import Path
from typing import List
from functools import wraps
import time
from contextlib import contextmanager
import unittest
import torch
from tokens.tokenizer import (
    MedicalTokenizer, 
    Config, 
    TokenizationUtilities,
    DatasetProcessor,
    GPUMemoryMonitor,
    MemoryMonitor
)

def with_retry(func, max_attempts=3, delay=1):
    """Retry a function with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempt = 0
        while attempt < max_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt == max_attempts:
                    raise e
                time.sleep(delay * (2 ** (attempt - 1)))
    return wrapper

@contextmanager
def managed_temp_file(suffix='.txt'):
    """Context manager for temporary file handling."""
    temp_path = Path(tempfile.mktemp(suffix=suffix))
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()

def split_chunk(chunk: List[str], num_parts: int):
    """Split a chunk into approximately equal parts."""
    chunk_size = len(chunk)
    part_size = max(1, chunk_size // num_parts)
    for i in range(0, chunk_size, part_size):
        yield chunk[i:min(i + part_size, chunk_size)] 

class TestMedicalTokenizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(local_data_path=self.temp_dir)
        self.tokenizer = MedicalTokenizer()

    def test_dynamic_vocab_size(self):
        # Test dynamic vocabulary sizing
        small_texts = ["test text"] * 100
        large_texts = ["test text"] * 10000
        
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('\n'.join(small_texts))
            f.flush()
            tokenizer_small = MedicalTokenizer()
            tokenizer_small.train([f.name], f"{self.temp_dir}/small_tokenizer.json")
            
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('\n'.join(large_texts))
            f.flush()
            tokenizer_large = MedicalTokenizer()
            tokenizer_large.train([f.name], f"{self.temp_dir}/large_tokenizer.json")
            
        self.assertLess(tokenizer_small.vocab_size, tokenizer_large.vocab_size)

    def test_padding(self):
        utils = TokenizationUtilities()
        input_ids = [torch.tensor([1, 2, 3]), torch.tensor([1, 2])]
        attention_masks = [torch.ones(3), torch.ones(2)]
        
        padded_ids, padded_masks = utils.dynamic_padding(input_ids, attention_masks)
        
        self.assertEqual(padded_ids.shape, (2, 3))
        self.assertEqual(padded_masks.shape, (2, 3))
        self.assertEqual(padded_masks[1][-1].item(), 0)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            TokenizationUtilities.dynamic_padding([], [])
            
        with self.assertRaises(PermissionError):
            Config("/root/invalid_path")  # Should fail on most systems

class TestDatasetProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(local_data_path=self.temp_dir)
        self.processor = DatasetProcessor([], self.config)

    def test_memory_monitoring(self):
        monitor = MemoryMonitor(threshold=0.0)  # Set to 0 to trigger immediately
        self.assertTrue(monitor.should_pause())
        
        monitor = MemoryMonitor(threshold=1.0)  # Set to 100% to never trigger
        self.assertFalse(monitor.should_pause())

    def test_chunk_generation(self):
        texts = ["test"] * 1000
        chunks = list(self.processor._adaptive_chunk_generator(texts, 100))
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    def test_batch_size_calculation(self):
        batch_size = self.processor._calculate_optimal_batch_size()
        self.assertTrue(100 <= batch_size <= 5000)

class TestGPUMemoryMonitor(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_monitoring(self):
        monitor = GPUMemoryMonitor(threshold=0.8)
        initial_state = monitor.should_pause()
        
        # Allocate some GPU memory
        tensor = torch.zeros(1000, 1000).cuda()
        after_allocation = monitor.should_pause()
        
        del tensor
        torch.cuda.empty_cache()
        after_cleanup = monitor.should_pause()
        
        self.assertIsInstance(initial_state, bool)
        self.assertIsInstance(after_allocation, bool)
        self.assertIsInstance(after_cleanup, bool)

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests() 