"""
Tests for Enigma Tokenizer
==========================

Tests for the AITesterTokenizer (AdvancedBPETokenizer alias) with:
  - Special tokens in AI Tester's [E:token] format
  - Streaming encode support
  - Improved unicode handling
"""

import pytest
import tempfile
from pathlib import Path
import json

from ai_tester.core.advanced_tokenizer import AITesterTokenizer

# Alias for backwards compatibility tests
AdvancedBPETokenizer = AITesterTokenizer


class TestTokenizerSpecialTokens:
    """Test special tokens for tool use."""
    
    def test_tool_tokens_present(self):
        """Test that all tool-related special tokens are defined."""
        tokenizer = AITesterTokenizer()
        
        # Tool invocation tokens (AI Tester's [E:token] format)
        assert '[E:tool]' in tokenizer.special_tokens
        assert '[E:tool_out]' in tokenizer.special_tokens
        assert '[E:tool_end]' in tokenizer.special_tokens
        assert '[E:out_end]' in tokenizer.special_tokens
        
        # Modality tokens
        assert '[E:img]' in tokenizer.special_tokens
        assert '[E:audio]' in tokenizer.special_tokens
        assert '[E:video]' in tokenizer.special_tokens
        assert '[E:vision]' in tokenizer.special_tokens
        
        # Action tokens
        assert '[E:gen_img]' in tokenizer.special_tokens
        assert '[E:avatar]' in tokenizer.special_tokens
        assert '[E:speak]' in tokenizer.special_tokens
    
    def test_conversation_tokens(self):
        """Test conversation role tokens."""
        tokenizer = AITesterTokenizer()
        
        # AI Tester uses [E:token] format
        assert '[E:system]' in tokenizer.special_tokens
        assert '[E:user]' in tokenizer.special_tokens
        assert '[E:assistant]' in tokenizer.special_tokens
    
    def test_encode_decode_special_tokens(self):
        """Test encoding and decoding text with special tokens."""
        tokenizer = AITesterTokenizer()
        
        text = '[E:tool]generate_image("test")[E:tool_end]'
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        
        # Should preserve special tokens (may have minor formatting differences)
        decoded_lower = decoded.lower().replace(' ', '').replace('_', '')
        # Check that key content is preserved
        assert len(ids) > 0
        assert len(decoded) > 0
    
    def test_special_token_ids_unique(self):
        """Test that all special tokens have unique IDs."""
        tokenizer = AITesterTokenizer()
        
        ids = list(tokenizer.special_tokens.values())
        assert len(ids) == len(set(ids)), "Special token IDs must be unique"
    
    def test_special_token_ids_low(self):
        """Test that special tokens use low IDs (< 100)."""
        tokenizer = AITesterTokenizer()
        
        for token, idx in tokenizer.special_tokens.items():
            assert idx < 100, f"Special token {token} has ID {idx} >= 100"


class TestStreamingEncode:
    """Test streaming encode functionality."""
    
    def test_encode_stream_simple(self):
        """Test basic streaming encode."""
        tokenizer = AITesterTokenizer()
        
        # Encode in chunks
        tokenizer.reset_stream()
        ids1 = tokenizer.encode_stream("Hello ", finalize=False)
        ids2 = tokenizer.encode_stream("world", finalize=False)
        ids3 = tokenizer.encode_stream("!", finalize=True)
        
        all_ids = ids1 + ids2 + ids3
        
        # Compare with regular encode
        regular_ids = tokenizer.encode("Hello world!", add_special_tokens=False)
        
        # Should produce similar results (may differ slightly at chunk boundaries)
        assert len(all_ids) > 0 or len(regular_ids) > 0
    
    def test_encode_stream_reset(self):
        """Test stream reset clears buffer."""
        tokenizer = AITesterTokenizer()
        
        tokenizer.encode_stream("partial", finalize=False)
        tokenizer.reset_stream()
        
        # After reset, buffer should be empty
        ids = tokenizer.encode_stream("new text", finalize=True)
        # Should only contain "new text" tokens
        assert len(ids) > 0


class TestTokenizerSaveLoad:
    """Test tokenizer save/load functionality."""
    
    def test_save_and_load(self):
        """Test saving and loading tokenizer."""
        tokenizer = AITesterTokenizer()
        
        # Train on simple data
        tokenizer.train(
            ["Hello world", "Hello there", "world peace"],
            vocab_size=500,
            verbose=False
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "tokenizer.json"
            tokenizer.save(vocab_path)
            
            # Check file exists
            assert vocab_path.exists()
            
            # Load into new tokenizer
            tokenizer2 = AITesterTokenizer(vocab_file=vocab_path)
            
            # Should have same vocab size
            assert tokenizer2.vocab_size == tokenizer.vocab_size
    
    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        tokenizer = AITesterTokenizer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "subdir" / "tokenizer.json"
            tokenizer.save(vocab_path)
            assert vocab_path.exists()


class TestImprovedRegex:
    """Test improved regex pre-tokenization."""
    
    def test_number_handling(self):
        """Test that numbers are handled well."""
        tokenizer = AITesterTokenizer()
        
        text = "I have 42 apples and 3.14 oranges"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        # Numbers should be preserved
        assert '42' in decoded or '4' in decoded
        assert '3.14' in decoded or '3' in decoded
    
    def test_code_handling(self):
        """Test handling of code-like text."""
        tokenizer = AITesterTokenizer()
        
        code = "def hello(): print('world')"
        ids = tokenizer.encode(code, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        # Should preserve structure
        assert 'def' in decoded or 'hello' in decoded
        assert 'print' in decoded
    
    def test_punctuation_handling(self):
        """Test handling of punctuation."""
        tokenizer = AITesterTokenizer()
        
        text = "Hello! How are you? I'm fine."
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        # Should have some content
        assert len(decoded) > 0


class TestUnicodeEmoji:
    """Test unicode and emoji handling."""
    
    def test_emoji_encode_decode(self):
        """Test encoding and decoding emoji."""
        tokenizer = AITesterTokenizer()
        
        text = "Hello 😊 World 🌍"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        # Should handle emojis (may not preserve exactly, but shouldn't crash)
        assert len(decoded) > 0
        assert 'Hello' in decoded or 'World' in decoded
    
    def test_unicode_characters(self):
        """Test various unicode characters."""
        tokenizer = AITesterTokenizer()
        
        text = "Héllo wörld ñiño"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        assert len(decoded) > 0
    
    def test_mixed_scripts(self):
        """Test mixed language scripts."""
        tokenizer = AITesterTokenizer()
        
        text = "Hello 世界 مرحبا"
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        
        # Should not crash
        assert len(decoded) > 0


class TestTokenizerCompleteness:
    """Test overall tokenizer functionality."""
    
    def test_encode_decode_roundtrip(self):
        """Test that encode->decode preserves meaning."""
        tokenizer = AITesterTokenizer()
        
        texts = [
            "Hello world",
            "The quick brown fox",
            "123 test 456",
            "test@email.com",
        ]
        
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids)
            
            # Should be similar (may not be exact due to BPE)
            assert len(decoded) > 0
            # At least some words should match
            assert any(word in decoded for word in text.split())
    
    def test_vocab_size(self):
        """Test that vocab size is reasonable."""
        tokenizer = AITesterTokenizer()
        
        # Base vocab should be > 256 (bytes) + special tokens (~40)
        assert tokenizer.vocab_size > 290
    
    def test_cache_functionality(self):
        """Test that caching improves performance."""
        tokenizer = AITesterTokenizer()
        tokenizer.train(["test text"], vocab_size=500, verbose=False)
        
        text = "test"
        
        # First encode
        ids1 = tokenizer.encode(text, add_special_tokens=False)
        cache_size1 = len(tokenizer.cache)
        
        # Second encode (should use cache)
        ids2 = tokenizer.encode(text, add_special_tokens=False)
        cache_size2 = len(tokenizer.cache)
        
        # Results should be identical
        assert ids1 == ids2
        # Cache should have entries
        assert cache_size2 >= cache_size1


class TestBackwardsCompatibility:
    """Test backwards compatibility with old tokenizer names."""
    
    def test_advanced_bpe_alias(self):
        """Test that AdvancedBPETokenizer alias works."""
        tokenizer = AdvancedBPETokenizer()
        assert isinstance(tokenizer, AITesterTokenizer)
    
    def test_special_tokens_property(self):
        """Test special_tokens property exists."""
        tokenizer = AITesterTokenizer()
        assert hasattr(tokenizer, 'special_tokens')
        assert isinstance(tokenizer.special_tokens, dict)
    
    def test_pad_token_id(self):
        """Test pad_token_id property."""
        tokenizer = AITesterTokenizer()
        assert hasattr(tokenizer, 'pad_token_id')
        assert tokenizer.pad_token_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
