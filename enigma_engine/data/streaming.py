"""
Dataset Streaming

Stream large datasets from disk or network without loading entirely into memory.
Supports shuffled streaming, batching, and preprocessing pipelines.

FILE: enigma_engine/data/streaming.py
TYPE: Data
MAIN CLASSES: StreamingDataset, ShuffleBuffer, DataPipeline
"""

import json
import logging
import mmap
import os
import queue
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Streaming configuration."""
    buffer_size: int = 10000  # Shuffle buffer size
    prefetch_batches: int = 2
    num_workers: int = 2
    seed: int = 42


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over data samples."""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples (approximate OK)."""


class TextFileSource(DataSource):
    """Stream text data from files (one sample per line)."""
    
    def __init__(self,
                 paths: list[Path],
                 encoding: str = "utf-8"):
        """
        Initialize text file source.
        
        Args:
            paths: List of file paths
            encoding: Text encoding
        """
        self.paths = [Path(p) for p in paths]
        self.encoding = encoding
        self._total_lines = None
    
    def __iter__(self) -> Iterator[str]:
        for path in self.paths:
            with open(path, encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    
    def __len__(self) -> int:
        if self._total_lines is None:
            self._total_lines = 0
            for path in self.paths:
                with open(path, encoding=self.encoding) as f:
                    self._total_lines += sum(1 for _ in f)
        return self._total_lines


class JSONLSource(DataSource):
    """Stream data from JSONL files."""
    
    def __init__(self, paths: list[Path]):
        """
        Initialize JSONL source.
        
        Args:
            paths: List of JSONL file paths
        """
        self.paths = [Path(p) for p in paths]
        self._total_samples = None
    
    def __iter__(self) -> Iterator[dict]:
        for path in self.paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
    
    def __len__(self) -> int:
        if self._total_samples is None:
            self._total_samples = 0
            for path in self.paths:
                with open(path) as f:
                    self._total_samples += sum(1 for _ in f)
        return self._total_samples


class MemoryMappedSource(DataSource):
    """Memory-mapped file source for efficient large file reading."""
    
    def __init__(self, path: Path, record_size: int):
        """
        Initialize memory-mapped source.
        
        Args:
            path: File path
            record_size: Fixed size of each record in bytes
        """
        self.path = Path(path)
        self.record_size = record_size
        self._file = None
        self._mmap = None
    
    def _open(self):
        if self._file is None:
            self._file = open(self.path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def __iter__(self) -> Iterator[bytes]:
        self._open()
        offset = 0
        file_size = self._mmap.size()
        
        while offset + self.record_size <= file_size:
            yield self._mmap[offset:offset + self.record_size]
            offset += self.record_size
    
    def __len__(self) -> int:
        return os.path.getsize(self.path) // self.record_size
    
    def __enter__(self):
        """Context manager entry - opens the file."""
        self._open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes resources."""
        self.close()
        return False
    
    def close(self):
        """Explicitly close the memory-mapped file and file handle."""
        try:
            if self._mmap:
                self._mmap.close()
                self._mmap = None
            if self._file:
                self._file.close()
                self._file = None
        except Exception:
            pass  # Ignore cleanup errors
    
    def __del__(self):
        try:
            if self._mmap:
                self._mmap.close()
            if self._file:
                self._file.close()
        except Exception:
            pass  # Ignore cleanup errors during shutdown


class ShuffleBuffer:
    """
    Reservoir-based streaming shuffle buffer.
    
    Maintains a buffer of samples and returns random samples
    while refilling from the stream.
    """
    
    def __init__(self, buffer_size: int = 10000, seed: int = None):
        """
        Initialize shuffle buffer.
        
        Args:
            buffer_size: Size of shuffle buffer
            seed: Random seed
        """
        self.buffer_size = buffer_size
        self.buffer: list[Any] = []
        self.rng = random.Random(seed)
    
    def add(self, item: Any) -> Optional[Any]:
        """
        Add item to buffer, potentially yielding a shuffled item.
        
        Args:
            item: Item to add
            
        Returns:
            Random item from buffer if full, else None
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(item)
            return None
        else:
            # Reservoir sampling: replace random element
            idx = self.rng.randint(0, len(self.buffer) - 1)
            out = self.buffer[idx]
            self.buffer[idx] = item
            return out
    
    def flush(self) -> Iterator[Any]:
        """Yield remaining items in shuffled order."""
        self.rng.shuffle(self.buffer)
        yield from self.buffer
        self.buffer = []


class DataPipeline:
    """
    Composable data transformation pipeline.
    
    Chain multiple transformations that are applied
    lazily during iteration.
    """
    
    def __init__(self, source: Union[DataSource, Iterator]):
        """
        Initialize pipeline.
        
        Args:
            source: Data source or iterator
        """
        self.source = source
        self.transforms: list[Callable] = []
    
    def map(self, fn: Callable) -> 'DataPipeline':
        """Apply function to each sample."""
        self.transforms.append(('map', fn))
        return self
    
    def filter(self, fn: Callable) -> 'DataPipeline':
        """Filter samples by predicate."""
        self.transforms.append(('filter', fn))
        return self
    
    def flat_map(self, fn: Callable) -> 'DataPipeline':
        """Apply function that returns multiple items."""
        self.transforms.append(('flat_map', fn))
        return self
    
    def batch(self, batch_size: int) -> 'DataPipeline':
        """Group samples into batches."""
        self.transforms.append(('batch', batch_size))
        return self
    
    def shuffle(self, buffer_size: int = 10000, seed: int = None) -> 'DataPipeline':
        """Shuffle samples using a buffer."""
        self.transforms.append(('shuffle', (buffer_size, seed)))
        return self
    
    def take(self, n: int) -> 'DataPipeline':
        """Take first n samples."""
        self.transforms.append(('take', n))
        return self
    
    def skip(self, n: int) -> 'DataPipeline':
        """Skip first n samples."""
        self.transforms.append(('skip', n))
        return self
    
    def __iter__(self) -> Iterator:
        """Iterate through transformed data."""
        stream = iter(self.source)
        
        for transform_type, param in self.transforms:
            if transform_type == 'map':
                stream = (param(x) for x in stream)
            
            elif transform_type == 'filter':
                stream = (x for x in stream if param(x))
            
            elif transform_type == 'flat_map':
                stream = (y for x in stream for y in param(x))
            
            elif transform_type == 'batch':
                stream = self._batch_iter(stream, param)
            
            elif transform_type == 'shuffle':
                buffer_size, seed = param
                stream = self._shuffle_iter(stream, buffer_size, seed)
            
            elif transform_type == 'take':
                stream = self._take_iter(stream, param)
            
            elif transform_type == 'skip':
                stream = self._skip_iter(stream, param)
        
        return stream
    
    @staticmethod
    def _batch_iter(stream: Iterator, batch_size: int) -> Iterator:
        """Batch iterator."""
        batch = []
        for item in stream:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    @staticmethod
    def _shuffle_iter(stream: Iterator, buffer_size: int, seed: int) -> Iterator:
        """Shuffle iterator."""
        buffer = ShuffleBuffer(buffer_size, seed)
        for item in stream:
            out = buffer.add(item)
            if out is not None:
                yield out
        yield from buffer.flush()
    
    @staticmethod
    def _take_iter(stream: Iterator, n: int) -> Iterator:
        """Take iterator."""
        count = 0
        for item in stream:
            if count >= n:
                break
            yield item
            count += 1
    
    @staticmethod
    def _skip_iter(stream: Iterator, n: int) -> Iterator:
        """Skip iterator."""
        count = 0
        for item in stream:
            if count >= n:
                yield item
            count += 1


class StreamingDataset:
    """
    PyTorch-compatible streaming dataset.
    
    Streams data from sources with automatic shuffling,
    batching, and prefetching.
    """
    
    def __init__(self,
                 source: DataSource,
                 config: StreamConfig = None,
                 transform: Callable = None):
        """
        Initialize streaming dataset.
        
        Args:
            source: Data source
            config: Streaming configuration
            transform: Optional transform function
        """
        self.source = source
        self.config = config or StreamConfig()
        self.transform = transform
        
        self._pipeline: Optional[DataPipeline] = None
        self._prefetch_queue: queue.Queue = None
        self._prefetch_thread: threading.Thread = None
        self._stop_prefetch = threading.Event()
    
    def _build_pipeline(self) -> DataPipeline:
        """Build data pipeline."""
        pipeline = DataPipeline(self.source)
        
        # Add shuffle
        if self.config.buffer_size > 0:
            pipeline = pipeline.shuffle(
                self.config.buffer_size,
                self.config.seed
            )
        
        # Add transform
        if self.transform:
            pipeline = pipeline.map(self.transform)
        
        return pipeline
    
    def _prefetch_worker(self, pipeline: DataPipeline, batch_size: int):
        """Background prefetching worker."""
        try:
            batch = []
            for item in pipeline:
                if self._stop_prefetch.is_set():
                    break
                
                batch.append(item)
                if len(batch) >= batch_size:
                    self._prefetch_queue.put(batch)
                    batch = []
            
            if batch:
                self._prefetch_queue.put(batch)
            
            self._prefetch_queue.put(None)  # Signal end
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
            self._prefetch_queue.put(None)
    
    def iter_batches(self, batch_size: int) -> Iterator[list]:
        """
        Iterate batches with prefetching.
        
        Args:
            batch_size: Batch size
            
        Yields:
            Batches of samples
        """
        pipeline = self._build_pipeline()
        
        # Setup prefetching
        self._prefetch_queue = queue.Queue(maxsize=self.config.prefetch_batches)
        self._stop_prefetch.clear()
        
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(pipeline, batch_size),
            daemon=True
        )
        self._prefetch_thread.start()
        
        try:
            while True:
                batch = self._prefetch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            self._stop_prefetch.set()
            if self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=1.0)
    
    def __iter__(self) -> Iterator:
        """Iterate samples one at a time."""
        pipeline = self._build_pipeline()
        return iter(pipeline)
    
    def __len__(self) -> int:
        """Return approximate dataset size."""
        return len(self.source)


class InterleaveDataset:
    """Interleave multiple streaming datasets."""
    
    def __init__(self,
                 datasets: list[StreamingDataset],
                 weights: list[float] = None,
                 seed: int = None):
        """
        Initialize interleaved dataset.
        
        Args:
            datasets: List of datasets to interleave
            weights: Sampling weights (None = uniform)
            seed: Random seed
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        self.rng = random.Random(seed)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def __iter__(self) -> Iterator:
        """Iterate with weighted interleaving."""
        iters = [iter(d) for d in self.datasets]
        active = list(range(len(iters)))
        
        while active:
            # Choose dataset by weight
            idx = self.rng.choices(
                active,
                weights=[self.weights[i] for i in active]
            )[0]
            
            try:
                yield next(iters[idx])
            except StopIteration:
                active.remove(idx)


def create_text_dataset(paths: list[str],
                        shuffle: bool = True,
                        buffer_size: int = 10000) -> StreamingDataset:
    """
    Create streaming text dataset.
    
    Args:
        paths: File paths
        shuffle: Enable shuffling
        buffer_size: Shuffle buffer size
        
    Returns:
        Streaming dataset
    """
    source = TextFileSource([Path(p) for p in paths])
    config = StreamConfig(
        buffer_size=buffer_size if shuffle else 0
    )
    return StreamingDataset(source, config)


def create_jsonl_dataset(paths: list[str],
                         shuffle: bool = True) -> StreamingDataset:
    """Create streaming JSONL dataset."""
    source = JSONLSource([Path(p) for p in paths])
    config = StreamConfig(buffer_size=10000 if shuffle else 0)
    return StreamingDataset(source, config)


__all__ = [
    'StreamingDataset',
    'StreamConfig',
    'DataSource',
    'TextFileSource',
    'JSONLSource',
    'MemoryMappedSource',
    'ShuffleBuffer',
    'DataPipeline',
    'InterleaveDataset',
    'create_text_dataset',
    'create_jsonl_dataset'
]
