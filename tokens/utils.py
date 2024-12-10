import tempfile
from pathlib import Path
from typing import List
from functools import wraps
import time
from contextlib import contextmanager

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