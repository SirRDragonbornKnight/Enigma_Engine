"""
Dataset Utilities for enigma_engine

Data loading, preprocessing, and tokenization helpers:
- Text dataset loading (JSON, JSONL, CSV, Parquet)
- Streaming dataset support
- Data preprocessing pipelines
- Tokenization utilities
- Dataset sharding for distributed training

Usage:
    from enigma_engine.core.datasets import TextDataset, DataLoader
    
    dataset = TextDataset.from_jsonl('data.jsonl', tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import csv
import json
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    max_length: int = 512
    truncation: bool = True
    padding: bool = True
    pad_token_id: int = 0
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    text_column: str = 'text'
    label_column: Optional[str] = None


class TextDataset(Dataset):
    """
    Dataset for text data.
    
    Supports multiple file formats and automatic tokenization.
    """
    
    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        labels: Optional[list[Any]] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self.labels = labels
        
        # Pre-tokenize if dataset is small enough
        self._cached_encodings = None
        if len(texts) < 10000:
            self._precompute_encodings()
    
    def _precompute_encodings(self):
        """Pre-compute all encodings."""
        self._cached_encodings = []
        
        for text in self.texts:
            encoding = self._tokenize(text)
            self._cached_encodings.append(encoding)
    
    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize a single text."""
        # Handle different tokenizer interfaces
        if hasattr(self.tokenizer, 'encode'):
            token_ids = self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, '__call__'):
            result = self.tokenizer(text)
            token_ids = result.get('input_ids', result)
        else:
            raise ValueError(f"Unknown tokenizer type: {type(self.tokenizer)}")
        
        # Convert to list if tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        # Truncate
        if self.config.truncation and len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        
        # Padding
        attention_mask = [1] * len(token_ids)
        
        if self.config.padding:
            pad_length = self.config.max_length - len(token_ids)
            token_ids = token_ids + [self.config.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
        }
        
        if self.config.return_attention_mask:
            result['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        
        return result
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cached_encodings:
            item = self._cached_encodings[idx].copy()
        else:
            item = self._tokenize(self.texts[idx])
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        
        return item
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        max_samples: Optional[int] = None
    ) -> 'TextDataset':
        """Load from JSONL file."""
        config = config or DataConfig()
        texts = []
        labels = []
        
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                item = json.loads(line)
                texts.append(item[config.text_column])
                
                if config.label_column and config.label_column in item:
                    labels.append(item[config.label_column])
        
        return cls(
            texts,
            tokenizer,
            config,
            labels if labels else None
        )
    
    @classmethod
    def from_json(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        max_samples: Optional[int] = None
    ) -> 'TextDataset':
        """Load from JSON file."""
        config = config or DataConfig()
        
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        texts = [item[config.text_column] for item in data]
        labels = None
        
        if config.label_column:
            labels = [item.get(config.label_column) for item in data]
        
        return cls(texts, tokenizer, config, labels)
    
    @classmethod
    def from_csv(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        max_samples: Optional[int] = None,
        delimiter: str = ','
    ) -> 'TextDataset':
        """Load from CSV file."""
        config = config or DataConfig()
        texts = []
        labels = []
        
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                
                texts.append(row[config.text_column])
                
                if config.label_column and config.label_column in row:
                    labels.append(row[config.label_column])
        
        return cls(texts, tokenizer, config, labels if labels else None)
    
    @classmethod
    def from_text_file(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        chunk_size: Optional[int] = None
    ) -> 'TextDataset':
        """Load from plain text file."""
        config = config or DataConfig()
        
        with open(path, encoding='utf-8') as f:
            text = f.read()
        
        if chunk_size:
            # Split into chunks
            texts = [
                text[i:i + chunk_size]
                for i in range(0, len(text), chunk_size)
            ]
        else:
            # Split by newlines
            texts = [line for line in text.split('\n') if line.strip()]
        
        return cls(texts, tokenizer, config)


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for large files.
    
    Processes data on-the-fly without loading everything into memory.
    """
    
    def __init__(
        self,
        paths: list[str],
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        shuffle_files: bool = True
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        self.shuffle_files = shuffle_files
    
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            paths = self.paths
        else:
            # Multi-process: shard files across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            paths = [
                p for i, p in enumerate(self.paths)
                if i % num_workers == worker_id
            ]
        
        if self.shuffle_files:
            paths = list(paths)
            random.shuffle(paths)
        
        for path in paths:
            yield from self._iterate_file(path)
    
    def _iterate_file(self, path: str) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over a single file."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.jsonl':
            with open(path, encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    text = item[self.config.text_column]
                    yield self._tokenize(text)
        
        elif ext == '.csv':
            with open(path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row[self.config.text_column]
                    yield self._tokenize(text)
        
        else:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield self._tokenize(line.strip())
    
    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize a single text."""
        if hasattr(self.tokenizer, 'encode'):
            token_ids = self.tokenizer.encode(text)
        else:
            token_ids = self.tokenizer(text)['input_ids']
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if self.config.truncation and len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        
        attention_mask = [1] * len(token_ids)
        
        if self.config.padding:
            pad_length = self.config.max_length - len(token_ids)
            token_ids = token_ids + [self.config.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class InstructionDataset(TextDataset):
    """
    Dataset for instruction-tuning (prompt, response pairs).
    """
    
    def __init__(
        self,
        instructions: list[dict[str, str]],
        tokenizer: Any,
        config: Optional[DataConfig] = None,
        prompt_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ):
        self.instructions = instructions
        self.prompt_template = prompt_template
        
        # Format texts
        texts = []
        for item in instructions:
            instruction = item.get('instruction', item.get('prompt', ''))
            response = item.get('response', item.get('output', ''))
            
            formatted = prompt_template.format(
                instruction=instruction,
                response=response
            )
            texts.append(formatted)
        
        super().__init__(texts, tokenizer, config)
    
    @classmethod
    def from_alpaca_format(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None
    ) -> 'InstructionDataset':
        """Load from Alpaca format (instruction, input, output)."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        
        instructions = []
        for item in data:
            instruction = item['instruction']
            if item.get('input'):
                instruction = f"{instruction}\n\nInput: {item['input']}"
            
            instructions.append({
                'instruction': instruction,
                'response': item['output']
            })
        
        return cls(instructions, tokenizer, config)
    
    @classmethod
    def from_sharegpt_format(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None
    ) -> 'InstructionDataset':
        """Load from ShareGPT format (conversations)."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        
        instructions = []
        for item in data:
            conversations = item.get('conversations', [])
            
            for i in range(0, len(conversations) - 1, 2):
                if i + 1 < len(conversations):
                    human = conversations[i]
                    assistant = conversations[i + 1]
                    
                    if human.get('from') in ('human', 'user'):
                        instructions.append({
                            'instruction': human.get('value', ''),
                            'response': assistant.get('value', '')
                        })
        
        return cls(instructions, tokenizer, config)


class PreferenceDataset(TextDataset):
    """
    Dataset for preference learning (DPO, RLHF).
    
    Contains (prompt, chosen, rejected) triplets.
    """
    
    def __init__(
        self,
        preferences: list[dict[str, str]],
        tokenizer: Any,
        config: Optional[DataConfig] = None
    ):
        self.preferences = preferences
        self.tokenizer = tokenizer
        self.config = config or DataConfig()
        
        self._cached_encodings = None
    
    def __len__(self) -> int:
        return len(self.preferences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.preferences[idx]
        
        prompt_encoding = self._tokenize(item['prompt'])
        chosen_encoding = self._tokenize(item['chosen'])
        rejected_encoding = self._tokenize(item['rejected'])
        
        return {
            'prompt_input_ids': prompt_encoding['input_ids'],
            'chosen_input_ids': chosen_encoding['input_ids'],
            'rejected_input_ids': rejected_encoding['input_ids'],
            'prompt_attention_mask': prompt_encoding.get('attention_mask'),
            'chosen_attention_mask': chosen_encoding.get('attention_mask'),
            'rejected_attention_mask': rejected_encoding.get('attention_mask'),
        }
    
    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize a single text."""
        if hasattr(self.tokenizer, 'encode'):
            token_ids = self.tokenizer.encode(text)
        else:
            token_ids = self.tokenizer(text)['input_ids']
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if self.config.truncation and len(token_ids) > self.config.max_length:
            token_ids = token_ids[:self.config.max_length]
        
        attention_mask = [1] * len(token_ids)
        
        if self.config.padding:
            pad_length = self.config.max_length - len(token_ids)
            token_ids = token_ids + [self.config.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: Any,
        config: Optional[DataConfig] = None
    ) -> 'PreferenceDataset':
        """Load from JSONL file with prompt, chosen, rejected columns."""
        preferences = []
        
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                preferences.append({
                    'prompt': item['prompt'],
                    'chosen': item['chosen'],
                    'rejected': item['rejected']
                })
        
        return cls(preferences, tokenizer, config)


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0
) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles variable-length sequences by padding.
    """
    # Get all keys
    keys = batch[0].keys()
    
    result = {}
    
    for key in keys:
        tensors = [item[key] for item in batch if item[key] is not None]
        
        if not tensors:
            continue
        
        # Find max length
        max_len = max(t.size(0) for t in tensors)
        
        # Pad and stack
        padded = []
        for t in tensors:
            if t.size(0) < max_len:
                padding = torch.full(
                    (max_len - t.size(0),),
                    pad_token_id if 'input_ids' in key else 0,
                    dtype=t.dtype
                )
                t = torch.cat([t, padding])
            padded.append(t)
        
        result[key] = torch.stack(padded)
    
    return result


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_token_id: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with proper collation.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        pin_memory=True
    )


def shard_dataset(
    dataset: TextDataset,
    num_shards: int,
    shard_id: int
) -> TextDataset:
    """
    Shard a dataset for distributed training.
    
    Args:
        dataset: Dataset to shard
        num_shards: Total number of shards
        shard_id: This shard's ID (0-indexed)
    
    Returns:
        Sharded dataset
    """
    indices = list(range(shard_id, len(dataset), num_shards))
    
    texts = [dataset.texts[i] for i in indices]
    labels = [dataset.labels[i] for i in indices] if dataset.labels else None
    
    return TextDataset(
        texts,
        dataset.tokenizer,
        dataset.config,
        labels
    )
