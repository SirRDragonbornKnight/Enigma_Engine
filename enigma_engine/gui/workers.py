"""
GUI Worker Threads - Background processing for the Enigma AI Engine GUI.

This module contains worker threads used by the GUI for background operations.
"""

import json
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
        self.is_gguf = getattr(engine, '_is_gguf', False) or (parent_window and getattr(parent_window, '_is_gguf_model', False))
        self.history = history
        self.system_prompt = system_prompt
        self.custom_tokenizer = custom_tokenizer
        self.parent_window = parent_window
        self._stop_requested = False
        self._start_time = None
    
    def stop(self):
        """Request the worker to stop generation."""
        self._stop_requested = True
    
    def _enhance_system_prompt_with_route(self):
        """Detect intent and enhance system prompt with route-specific instructions."""
        try:
            from ..core.tool_router import get_route_prompt, ROUTING_RULES
            
            # Simple keyword-based intent detection
            text_lower = self.text.lower()
            detected_intent = "chat"  # Default
            
            # Check each route's keywords
            for route_name, rule in ROUTING_RULES.items():
                for keyword in rule.keywords:
                    if keyword.lower() in text_lower:
                        detected_intent = route_name
                        break
                if detected_intent != "chat":
                    break
            
            # Get the route-specific prompt
            route_prompt = get_route_prompt(detected_intent, self.system_prompt)
            
            if route_prompt and route_prompt != self.system_prompt:
                self.system_prompt = route_prompt
                if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                    self.parent_window.log_terminal(f"Using {detected_intent} route prompt", "debug")
        except Exception as e:
            # If route detection fails, just use original prompt
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"Route detection skipped: {e}", "debug")
        
    def run(self):
        """Execute the AI generation in background thread."""
        try:
            import time
            self._start_time = time.time()
            
            self.thinking.emit("Analyzing your message...")
            
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"NEW REQUEST: {self.text}", "info")
            
            # Detect intent and get route-specific prompt
            self._enhance_system_prompt_with_route()
            
            if self._stop_requested:
                self.stopped.emit()
                return
            
            # Try unified inference for consistent tool handling across ALL models
            use_unified = getattr(self.parent_window, '_use_unified_inference', True) if self.parent_window else True
            
            if use_unified:
                response = self._generate_unified()
            elif self.is_gguf:
                response = self._generate_gguf()
            elif self.is_hf:
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
    
    def _generate_unified(self) -> str:
        """
        Generate response using UnifiedEngine - works identically for ALL model types.
        
        This ensures:
        - GGUF models, HuggingFace models, and Enigma models all behave the same
        - Tool calling works consistently
        - Agent loop (multi-step tool execution) works for all
        """
        self.thinking.emit("Using unified inference...")
        
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal("Unified inference - same behavior for all model types", "info")
        
        try:
            from ..core.unified_inference import UnifiedEngine
            
            # Determine model type
            if self.is_gguf:
                model_type = "gguf"
                model = self.engine.model
            elif self.is_hf:
                model_type = "huggingface"
                model = self.engine.model
            else:
                model_type = "enigma"
                model = self.engine
            
            # Create unified engine
            unified = UnifiedEngine(
                model=model,
                model_type=model_type,
                enable_agent_loop=True,
                keyword_routing=True
            )
            
            # Build history
            history = []
            if self.parent_window and hasattr(self.parent_window, 'chat_messages'):
                recent = self.parent_window.chat_messages[-6:-1] if len(self.parent_window.chat_messages) > 1 else []
                for msg in recent:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    history.append({"role": role, "content": msg.get("text", "")})
            
            if self._stop_requested:
                return ""
            
            self.thinking.emit("Running unified inference with tool support...")
            
            # Generate with tools
            result = unified.generate_with_tools(
                user_message=self.text,
                history=history,
                system_prompt=self.system_prompt,
                agent_loop=True,
                max_iterations=3,  # Allow up to 3 tool iterations
                max_tokens=512,
                temperature=0.7
            )
            
            response = result.get('response', '')
            tool_calls = result.get('tool_calls', [])
            tool_results = result.get('tool_results', [])
            iterations = result.get('iterations', 0)
            
            # Log what happened
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                if tool_calls:
                    self.parent_window.log_terminal(
                        f"Unified inference: {len(tool_calls)} tool call(s), {iterations} iteration(s)", 
                        "info"
                    )
                    for tc in tool_calls:
                        self.parent_window.log_terminal(f"  - {tc.get('tool', 'unknown')}", "debug")
            
            # Format tool calls for the existing execution pipeline
            if tool_calls and not tool_results:
                # Tools detected but not executed by UnifiedEngine - add to response for GUI execution
                tool_call_strs = []
                for tc in tool_calls:
                    tool_call_strs.append(
                        f'<tool_call>{{"tool": "{tc.get("tool", "")}", "params": {json.dumps(tc.get("params", {}))}}}</tool_call>'
                    )
                response = f"{response}\n\n{''.join(tool_call_strs)}"
            
            return response if response else "(No response generated)"
            
        except Exception as e:
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"Unified inference failed, falling back: {e}", "warning")
            
            # Fallback to model-specific generation
            if self.is_gguf:
                return self._generate_gguf()
            elif self.is_hf:
                return self._generate_hf()
            else:
                return self._generate_local()

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
                max_new_tokens=512,  # Increased for tool calls
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
    
    def _generate_gguf(self) -> str:
        """Generate response using GGUF model (llama.cpp) with native function calling."""
        self.thinking.emit("Building conversation context...")
        
        if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
            self.parent_window.log_terminal("GGUF model - using llama.cpp with native tool support...", "info")
        
        if self._stop_requested:
            return ""
        
        # Get the GGUF model from the engine
        gguf_model = self.engine.model
        
        # Build chat history for context
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        if self.parent_window and hasattr(self.parent_window, 'chat_messages'):
            recent = self.parent_window.chat_messages[-6:-1] if len(self.parent_window.chat_messages) > 1 else []
            for msg in recent:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("text", "")})
        
        # Add current message
        messages.append({"role": "user", "content": self.text})
        
        self.thinking.emit("Running GGUF inference with tool support...")
        
        if self._stop_requested:
            return ""
        
        try:
            # Try native function calling first (much more reliable)
            if hasattr(gguf_model, 'chat_with_tools'):
                self.thinking.emit("Using native function calling...")
                result = gguf_model.chat_with_tools(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7
                )
                
                response_text = result.get('content', '')
                tool_calls = result.get('tool_calls', [])
                
                # Process any tool calls the AI made
                if tool_calls:
                    if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                        self.parent_window.log_terminal(f"AI made {len(tool_calls)} native tool call(s)", "info")
                    
                    # Format tool calls for execution
                    tool_results = []
                    for tc in tool_calls:
                        func = tc.get('function', {})
                        tool_name = func.get('name', '')
                        
                        # Parse arguments - could be string or dict
                        args = func.get('arguments', '{}')
                        if isinstance(args, str):
                            import json
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        
                        # Convert to our <tool_call> format for unified execution
                        tool_call_str = f'<tool_call>{{"tool": "{tool_name}", "params": {json.dumps(args)}}}</tool_call>'
                        tool_results.append(tool_call_str)
                    
                    # Append tool calls to response for unified processing
                    if tool_results:
                        response_text = f"{response_text}\n\n{''.join(tool_results)}"
                
                return response_text
                
            # Fallback to regular chat
            elif hasattr(gguf_model, 'chat'):
                response = gguf_model.chat(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.7
                )
                return response if isinstance(response, str) else str(response)
            else:
                # Fall back to generate
                prompt = f"User: {self.text}\nAssistant:"
                response = gguf_model.generate(
                    prompt,
                    max_tokens=512,
                    temperature=0.7
                )
                return response if isinstance(response, str) else str(response)
            
        except Exception as e:
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"GGUF generation error: {e}", "error")
            return f"Error generating response: {e}"
            
        except Exception as e:
            if self.parent_window and hasattr(self.parent_window, 'log_terminal'):
                self.parent_window.log_terminal(f"GGUF generation error: {e}", "error")
            return f"Error generating response: {e}"
    
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
        
        if hasattr(self.engine, 'chat'):
            # Use chat method with system prompt support
            response = self.engine.chat(
                message=self.text,
                history=chat_history if chat_history else None,
                system_prompt=self.system_prompt,
                max_gen=100,
                auto_truncate=True
            )
            formatted_prompt = self.text
        else:
            # Fallback: Build prompt with system prompt if available
            if self.system_prompt:
                formatted_prompt = f"System: {self.system_prompt}\n\nQ: {self.text}\nA:"
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
