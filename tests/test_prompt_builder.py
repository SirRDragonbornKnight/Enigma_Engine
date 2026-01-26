"""
Tests for the centralized prompt builder module.
"""

import unittest
import tempfile
import json
from pathlib import Path


class TestPromptTemplate(unittest.TestCase):
    """Test PromptTemplate dataclass."""
    
    def test_default_template(self):
        """Default template should have expected values."""
        from forge_ai.core.prompt_builder import PromptTemplate
        
        t = PromptTemplate()
        self.assertEqual(t.system_prefix, "System: ")
        self.assertEqual(t.user_prefix, "User: ")
        self.assertEqual(t.assistant_prefix, "Assistant: ")
        self.assertTrue(t.add_generation_prefix)
    
    def test_template_to_dict(self):
        """Template should serialize to dict."""
        from forge_ai.core.prompt_builder import PromptTemplate
        
        t = PromptTemplate(user_prefix="Q: ", assistant_prefix="A: ")
        d = t.to_dict()
        
        self.assertIsInstance(d, dict)
        self.assertEqual(d["user_prefix"], "Q: ")
        self.assertEqual(d["assistant_prefix"], "A: ")
    
    def test_template_from_dict(self):
        """Template should deserialize from dict."""
        from forge_ai.core.prompt_builder import PromptTemplate
        
        d = {"user_prefix": "Human: ", "assistant_prefix": "Bot: "}
        t = PromptTemplate.from_dict(d)
        
        self.assertEqual(t.user_prefix, "Human: ")
        self.assertEqual(t.assistant_prefix, "Bot: ")


class TestPromptBuilder(unittest.TestCase):
    """Test PromptBuilder class."""
    
    def test_builder_creation(self):
        """Should create builder with default template."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        self.assertIsNotNone(builder.template)
    
    def test_build_simple_prompt(self):
        """Should build a simple chat prompt."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        prompt = builder.build_chat_prompt("Hello!")
        
        self.assertIn("User: Hello!", prompt)
        self.assertIn("Assistant:", prompt)
    
    def test_build_prompt_with_system(self):
        """Should include system prompt."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        prompt = builder.build_chat_prompt(
            "Hello!",
            system_prompt="You are helpful."
        )
        
        self.assertIn("System: You are helpful.", prompt)
        self.assertIn("User: Hello!", prompt)
    
    def test_build_prompt_with_history(self):
        """Should include conversation history."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt = builder.build_chat_prompt("How are you?", history=history)
        
        self.assertIn("User: Hi", prompt)
        self.assertIn("Assistant: Hello!", prompt)
        self.assertIn("User: How are you?", prompt)
    
    def test_extract_response_simple(self):
        """Should extract AI response from generated text."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        output = "User: Hello!\nAssistant: Hi there! How can I help?"
        
        response = builder.extract_response(output)
        self.assertEqual(response, "Hi there! How can I help?")
    
    def test_extract_response_with_stop_sequence(self):
        """Should stop at stop sequences."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        output = "Assistant: Hello!\nUser: Another message"
        
        response = builder.extract_response(output)
        self.assertEqual(response, "Hello!")
    
    def test_get_stop_sequences(self):
        """Should return stop sequences for current template."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        stops = builder.get_stop_sequences()
        
        self.assertIsInstance(stops, list)
        self.assertIn("\nUser:", stops)
    
    def test_set_template(self):
        """Should allow changing template."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder("default")
        builder.set_template("simple")
        
        prompt = builder.build_chat_prompt("Test")
        self.assertIn("Q:", prompt)
        self.assertIn("A:", prompt)
    
    def test_format_training_example(self):
        """Should format training examples correctly."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        example = builder.format_training_example(
            "Hello",
            "Hi there!"
        )
        
        self.assertIn("User: Hello", example)
        self.assertIn("Assistant: Hi there!", example)


class TestPresetTemplates(unittest.TestCase):
    """Test preset templates."""
    
    def test_chatml_template(self):
        """ChatML template should use special tokens."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder("chatml")
        prompt = builder.build_chat_prompt("Test")
        
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
    
    def test_simple_template(self):
        """Simple template should use Q/A format."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder("simple")
        prompt = builder.build_chat_prompt("Test")
        
        self.assertIn("Q: Test", prompt)
        self.assertIn("A:", prompt)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_get_prompt_builder_singleton(self):
        """get_prompt_builder should return same instance."""
        from forge_ai.core.prompt_builder import get_prompt_builder
        
        b1 = get_prompt_builder()
        b2 = get_prompt_builder()
        
        self.assertIs(b1, b2)
    
    def test_build_chat_prompt_function(self):
        """Convenience function should work."""
        from forge_ai.core.prompt_builder import build_chat_prompt
        
        prompt = build_chat_prompt("Hello")
        
        self.assertIn("User: Hello", prompt)
        self.assertIn("Assistant:", prompt)
    
    def test_extract_response_function(self):
        """Convenience function should work."""
        from forge_ai.core.prompt_builder import extract_response
        
        response = extract_response("Assistant: Test response")
        
        self.assertEqual(response, "Test response")


class TestRoleNormalization(unittest.TestCase):
    """Test that different role names are handled correctly."""
    
    def test_human_role_normalized(self):
        """'human' role should be treated as 'user'."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        history = [{"role": "human", "content": "Hi"}]
        prompt = builder.build_chat_prompt("Test", history=history)
        
        self.assertIn("User: Hi", prompt)
    
    def test_bot_role_normalized(self):
        """'bot' role should be treated as 'assistant'."""
        from forge_ai.core.prompt_builder import PromptBuilder
        
        builder = PromptBuilder()
        history = [{"role": "bot", "content": "Hello"}]
        prompt = builder.build_chat_prompt("Test", history=history)
        
        self.assertIn("Assistant: Hello", prompt)


if __name__ == "__main__":
    unittest.main()
