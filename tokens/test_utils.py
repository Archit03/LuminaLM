import unittest
import time
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils import (
    with_retry,
    managed_temp_file,
    MemoryManager,
    ChunkManager
)

class TestRetryDecorator(unittest.TestCase):
    def test_successful_execution(self):
        @with_retry
        def success_func():
            return "success"
        
        self.assertEqual(success_func(), "success")

    def test_retry_on_failure(self):
        self.attempt = 0
        
        @with_retry(max_attempts=3, delay=0.1)
        def fail_then_succeed():
            self.attempt += 1
            if self.attempt < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        self.assertEqual(fail_then_succeed(), "success")
        self.assertEqual(self.attempt, 2)

    def test_max_attempts_reached(self):
        @with_retry(max_attempts=3, delay=0.1)
        def always_fail():
            raise ValueError("Persistent failure")
        
        with self.assertRaises(ValueError):
            always_fail()


class TestManagedTempFile(unittest.TestCase):
    def test_temp_file_creation_and_cleanup(self):
        with managed_temp_file(suffix='.txt') as temp_path:
            self.assertTrue(temp_path.exists())
            self.assertTrue(str(temp_path).endswith('.txt'))
            
            # Write some data
            temp_path.write_text("test data")
            self.assertEqual(temp_path.read_text(), "test data")
            
        # File should be deleted after context
        self.assertFalse(temp_path.exists())


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.memory_manager = MemoryManager(
            gpu_memory_threshold=0.8,
            cpu_memory_threshold=0.85
        )

    @patch('psutil.virtual_memory')
    def test_should_reduce_batch(self, mock_vm):
        mock_vm.return_value = MagicMock(percent=90)  # 90% CPU usage
        self.assertTrue(self.memory_manager.should_reduce_batch())
        
        mock_vm.return_value = MagicMock(percent=70)  # 70% CPU usage
        self.assertFalse(self.memory_manager.should_reduce_batch())

    def test_get_optimal_batch_size(self):
        current_size = 1000
        
        # When memory usage is low
        with patch.object(self.memory_manager, 'should_reduce_batch', return_value=False):
            self.assertEqual(
                self.memory_manager.get_optimal_batch_size(current_size),
                current_size
            )
        
        # When memory usage is high
        with patch.object(self.memory_manager, 'should_reduce_batch', return_value=True):
            self.assertEqual(
                self.memory_manager.get_optimal_batch_size(current_size),
                500  # Should be reduced by factor of 0.5
            )


class TestChunkManager(unittest.TestCase):
    def setUp(self):
        self.memory_manager = MemoryManager()
        self.chunk_manager = ChunkManager(self.memory_manager)

    def test_get_chunk_size(self):
        # Small data size
        self.assertEqual(self.chunk_manager.get_chunk_size(500), 500)
        
        # Large data size with low memory pressure
        with patch.object(self.memory_manager, 'should_reduce_batch', return_value=False):
            self.assertGreater(self.chunk_manager.get_chunk_size(10000), 1000)

    def test_chunk_iterator(self):
        data = list(range(1000))
        chunk_size = 100
        
        chunks = list(self.chunk_manager.chunk_iterator(data, chunk_size))
        
        self.assertEqual(len(chunks), 10)
        self.assertEqual(len(chunks[0]), chunk_size)
        self.assertEqual(sum(len(chunk) for chunk in chunks), len(data))


if __name__ == '__main__':
    unittest.main() 