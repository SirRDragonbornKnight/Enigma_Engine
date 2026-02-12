"""
GUI Worker Threads - Background processing for the Enigma AI Engine GUI.

This module contains worker threads used by the GUI for background operations.
"""

from PyQt5.QtCore import QThread, pyqtSignal


class AIGenerationWorker(QThread):
    """Background worker for AI generation to keep GUI responsive.
    
    Signals:
        finished(str): Emits the response text when generation completes
        error(str): Emits error message if generation fails
        thinking(str): Emits thinking/reasoning status updates
        stopped(): Emits when generation is stopped by user
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    thinking = pyqtSignal(str)
    stopped = pyqtSignal()
    
    def __init__(self, engine, text, is_hf, history=None, system_prompt=None, 
                 custom_tokenizer=None, parent_window=None):
        """Initialize the AI generation worker.
        
        Args:
            engine: The inference engine (EnigmaEngine or HuggingFace model)
            text: User input text to generate response for
            is_hf: True if using HuggingFace model, False for local Forge
            history: Optional conversation history
            system_prompt: Optional system prompt
            custom_tokenizer: Optional custom tokenizer
            parent_window: Reference to parent window for logging
        """
        super().__init__()
        self.engine = engine
        self.text = text
        self.is_hf = is_hf
        self.history = history
        self.system_prompt = system_prompt
        self.custom_tokenizer = custom_tokenizer
        self.parent_window = parent_window
        self._stop_requested = False
        self._start_time = None
    
    def stop(self):
        """Request the worker to stop generation."""
        self._stop_requested = True
        
    def run(self):
        """Execute the AI generation in background thread."""
        try:
            import time
            self._start_time = time.time()
            
            self.thinking.emit("Analyzing your message...")
            
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"NEW REQUEST: {self.text}", "info")
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            if self.is_hf:
                response = self._generate_hf()
            else:
                response = self._generate_local()
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            if not response:
                response = "(No response generated - model may need more training)"
            
            response = self._validate_response(response)
            
            elapsed = time.time() - self._start_time
            self.thinking.emit(f"Done in {elapsed:.1f}s")
            self.finished.emit(response)
            
        except Exception as e:
            if self._stop_requested:
                self.stopped.emit()
            else:
                self.error.emit(str(e))
    
    def _generate_hf(self) -> str:
        """Generate response using HuggingFace model."""
        import time
        
        self.thinking.emit("Building conversation context...")
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal("Building conversation history...", "debug")
        time.sleep(0.1)
        
        if self._stop_requested:
            return ""
        
        self.thinking.emit("Processing with language model...")
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal("Running inference on model...", "info")
        
        if hasattr(self.engine.model, 'chat') and not self.custom_tokenizer:
            response = self.engine.model.chat(
                self.text,
                history=self.history if self.history else None,
                system_prompt=self.system_prompt,
                max_new_tokens=200,
                temperature=0.7
            )
        else:
            self.thinking.emit("Tokenizing input...")
            response = self.engine.model.generate(
                self.text,
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                custom_tokenizer=self.custom_tokenizer
            )
        
        return self._decode_response(response)
    
    def _generate_local(self) -> str:
        """Generate response using local Forge model."""
        self.thinking.emit("Building conversation context...")
        
        chat_history = []
        if self.parent_window and hasattr(self.parent_window, 'chat_messages'):
            recent = self.parent_window.chat_messages[-7:-1] if len(self.parent_window.chat_messages) > 1 else []
            for msg in recent:
                role = "user" if msg.get("role") == "user" else "assistant"
                chat_history.append({"role": role, "content": msg.get("text", "")})
        
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal(f"Using {len(chat_history)} history messages", "debug")
        
        if self._stop_requested:
            return ""
        
        self.thinking.emit("Running inference on local model...")
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal("Generating tokens...", "info")
        
        if hasattr(self.engine, 'chat') and chat_history:
            response = self.engine.chat(
                message=self.text,
                history=chat_history,
                system_prompt=self.system_prompt,
                max_gen=100,
                auto_truncate=True
            )
            formatted_prompt = self.text
        else:
            formatted_prompt = f"Q: {self.text}\nA:"
            response = self.engine.generate(formatted_prompt, max_gen=100)
        
        if self._stop_requested:
            return ""
        
        self.thinking.emit("Cleaning up response...")
        return self._clean_response(response, formatted_prompt)
    
    def _decode_response(self, response) -> str:
        """Decode tensor response to text if needed."""
        if not (hasattr(response, 'shape') or 'tensor' in str(type(response)).lower()):
            return response
        
        self.thinking.emit("Decoding model output...")
        try:
            import torch
            if isinstance(response, torch.Tensor):
                if hasattr(self.engine.model, 'tokenizer'):
                    return self.engine.model.tokenizer.decode(
                        response[0] if len(response.shape) > 1 else response,
                        skip_special_tokens=True
                    )
                elif self.custom_tokenizer:
                    return self.custom_tokenizer.decode(
                        response[0] if len(response.shape) > 1 else response,
                        skip_special_tokens=True
                    )
        except Exception as e:
            return f"[Warning] Could not decode model output: {e}"
        
        return (
            "[Warning] Model returned raw tensor data. This usually means:\n"
            "  The model is not properly configured for text generation\n"
            "  Try a different model or check if it needs fine-tuning"
        )
    
    def _clean_response(self, response: str, formatted_prompt: str) -> str:
        """Clean up the response text."""
        if hasattr(response, 'shape') or 'tensor' in str(type(response)).lower():
            return (
                "[Warning] Model returned raw data instead of text.\n"
                "This model may need more training. Go to the Train tab."
            )
        
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):].strip()
        elif response.startswith(self.text):
            response = response[len(self.text):].strip()
            
        if "\nQ:" in response:
            response = response.split("\nQ:")[0].strip()
        if "Q:" in response:
            response = response.split("Q:")[0].strip()
        if response.startswith("A:"):
            response = response[2:].strip()
        if response.startswith(":"):
            response = response[1:].strip()
        
        return response
    
    def _validate_response(self, response: str) -> str:
        """Validate response and detect garbage/code output."""
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal(f"Generated {len(response)} characters", "success")
        
        garbage_indicators = [
            'torch.tensor', 'np.array', 'def test_', 'assert ', 'import torch',
            'class Test', 'self.setup', '.to(device)', 'cudnn.enabled',
            'torch.randn', 'torch.zeros', 'return Tensor', '# Convert',
            'dtype=torch.float', 'skip_special_tokens', "'cuda:0'", "'cuda:1'",
            '.to("cuda', 'tensor([[', 'Output:', '# Output:', 'tensors.shape',
            '.expand(', '```python', 'import torch', 'broadcasting dimension',
            'tensor(tensor(', '.size() ==', 'expanded_matrix'
        ]
        
        is_garbage = any(indicator in response for indicator in garbage_indicators)
        
        if not is_garbage and len(response) > 50:
            code_chars = response.count('(') + response.count(')') + response.count('[') + response.count(']') + response.count('=')
            is_garbage = code_chars > len(response) * 0.1
        
        if is_garbage:
            return (
                "[Warning] The model generated code/technical text instead of a response.\n\n"
                "This can happen with small models. Try:\n"
                "  Using a larger model (tinyllama_chat, phi2, qwen2_1.5b_instruct)\n"
                "  Being more specific in your question\n"
                "  Training your own Forge model with conversational data"
            )
        
        return response
