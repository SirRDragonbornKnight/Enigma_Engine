"""
Built-in Embeddings

Zero-dependency text embeddings using:
1. TF-IDF style word frequency vectors
2. Character n-gram hashing
3. Simple bag-of-words

These won't be as good as sentence-transformers but work without any installs.
"""

import hashlib
import math
import re
import time
from collections import Counter
from typing import Dict, Any, List, Optional


class BuiltinEmbeddings:
    """
    Built-in embedding generator - no external dependencies.
    
    Uses a combination of:
    - Word frequency (TF-IDF inspired)
    - Character n-grams for subword features
    - Hash-based dimensionality reduction
    """
    
    def __init__(self, dimensions: int = 384):
        """
        Initialize embeddings.
        
        Args:
            dimensions: Output embedding size (default 384 to match MiniLM)
        """
        self.dimensions = dimensions
        self.is_loaded = False
        self._word_idf: Dict[str, float] = {}
        self._vocab_size = 0
        
    def load(self) -> bool:
        """Load the embedding model (always succeeds)."""
        # Pre-compute some common English word IDFs
        # Higher IDF = rarer word = more important
        common_words = {
            'the': 0.1, 'a': 0.1, 'an': 0.1, 'is': 0.2, 'are': 0.2,
            'was': 0.2, 'were': 0.2, 'be': 0.2, 'been': 0.3, 'being': 0.3,
            'have': 0.3, 'has': 0.3, 'had': 0.3, 'do': 0.3, 'does': 0.3,
            'did': 0.3, 'will': 0.3, 'would': 0.4, 'could': 0.4, 'should': 0.4,
            'may': 0.4, 'might': 0.4, 'must': 0.4, 'shall': 0.5,
            'to': 0.1, 'of': 0.1, 'in': 0.2, 'for': 0.2, 'on': 0.2,
            'with': 0.2, 'at': 0.3, 'by': 0.3, 'from': 0.3, 'as': 0.3,
            'into': 0.4, 'through': 0.4, 'during': 0.5, 'before': 0.5,
            'after': 0.5, 'above': 0.5, 'below': 0.5, 'between': 0.5,
            'and': 0.1, 'or': 0.2, 'but': 0.3, 'if': 0.3, 'because': 0.5,
            'this': 0.2, 'that': 0.2, 'these': 0.3, 'those': 0.3,
            'it': 0.2, 'its': 0.3, 'i': 0.2, 'you': 0.2, 'he': 0.3,
            'she': 0.3, 'we': 0.3, 'they': 0.3, 'my': 0.3, 'your': 0.3,
            'his': 0.3, 'her': 0.3, 'our': 0.4, 'their': 0.4,
            'what': 0.3, 'which': 0.3, 'who': 0.3, 'when': 0.4, 'where': 0.4,
            'why': 0.4, 'how': 0.4, 'all': 0.3, 'each': 0.4, 'every': 0.4,
            'both': 0.4, 'few': 0.5, 'more': 0.4, 'most': 0.4, 'other': 0.4,
            'some': 0.4, 'such': 0.5, 'no': 0.3, 'not': 0.2, 'only': 0.4,
            'same': 0.5, 'so': 0.3, 'than': 0.4, 'too': 0.4, 'very': 0.4,
            'just': 0.4, 'also': 0.4, 'now': 0.4, 'here': 0.5, 'there': 0.4,
        }
        self._word_idf = common_words
        self.is_loaded = True
        return True
    
    def unload(self):
        """Unload (no-op for built-in)."""
        self.is_loaded = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens
    
    def _get_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Get character n-grams."""
        text = text.lower()
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams
    
    def _hash_to_index(self, item: str, seed: int = 0) -> int:
        """Hash a string to an index in the embedding space."""
        h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
        return int(h, 16) % self.dimensions
    
    def _get_idf(self, word: str) -> float:
        """Get IDF weight for a word (higher = rarer = more important)."""
        return self._word_idf.get(word, 1.0)  # Default 1.0 for unknown words
    
    def embed(self, text: str) -> Dict[str, Any]:
        """Generate embedding for a single text."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        if not text.strip():
            return {"success": False, "error": "Empty text"}
        
        try:
            start = time.time()
            
            # Initialize embedding vector
            embedding = [0.0] * self.dimensions
            
            # 1. Word-level features (TF-IDF style)
            tokens = self._tokenize(text)
            word_counts = Counter(tokens)
            total_words = len(tokens) or 1
            
            for word, count in word_counts.items():
                tf = count / total_words
                idf = self._get_idf(word)
                weight = tf * idf
                
                # Hash word to multiple positions for better coverage
                for seed in range(3):
                    idx = self._hash_to_index(word, seed)
                    embedding[idx] += weight * (1.0 / (seed + 1))
            
            # 2. Character n-gram features
            ngrams = self._get_ngrams(text, 3)
            ngram_counts = Counter(ngrams)
            total_ngrams = len(ngrams) or 1
            
            for ngram, count in ngram_counts.items():
                weight = count / total_ngrams * 0.5  # Weight ngrams less than words
                idx = self._hash_to_index(f"ng:{ngram}", 0)
                embedding[idx] += weight
            
            # 3. Normalize to unit vector
            magnitude = math.sqrt(sum(x*x for x in embedding))
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
            
            return {
                "success": True,
                "embedding": embedding,
                "dimensions": self.dimensions,
                "duration": time.time() - start
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def embed_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Generate embeddings for multiple texts."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            start = time.time()
            embeddings = []
            
            for text in texts:
                result = self.embed(text)
                if result["success"]:
                    embeddings.append(result["embedding"])
                else:
                    # Use zero vector for failed embeddings
                    embeddings.append([0.0] * self.dimensions)
            
            return {
                "success": True,
                "embeddings": embeddings,
                "dimensions": self.dimensions,
                "count": len(embeddings),
                "duration": time.time() - start
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate cosine similarity between two texts."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}
        
        try:
            start = time.time()
            
            result1 = self.embed(text1)
            result2 = self.embed(text2)
            
            if not result1["success"] or not result2["success"]:
                return {"success": False, "error": "Failed to embed texts"}
            
            emb1 = result1["embedding"]
            emb2 = result2["embedding"]
            
            # Cosine similarity (vectors are already normalized)
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            
            # Clamp to [-1, 1] to handle floating point errors
            similarity = max(-1.0, min(1.0, dot_product))
            
            return {
                "success": True,
                "similarity": similarity,
                "duration": time.time() - start
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
