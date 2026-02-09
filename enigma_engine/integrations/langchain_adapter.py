"""
LangChain Integration

Adapters for using Enigma AI Engine with LangChain framework.
Provides LLM, Chat Model, and Embeddings implementations.

FILE: enigma_engine/integrations/langchain_adapter.py
TYPE: Integration
MAIN CLASSES: ForgeLLM, ForgeChatModel, ForgeEmbeddings
"""

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for LangChain availability
try:
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import LLM
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # Create stub classes for type hints
    class LLM:
        pass
    class BaseChatModel:
        pass
    class Embeddings:
        pass


@dataclass
class ForgeModelConfig:
    """Configuration for Forge LangChain models."""
    model_name: str = "forge-ai"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 50
    stop_sequences: list[str] = None
    streaming: bool = False


if HAS_LANGCHAIN:
    
    class ForgeLLM(LLM):
        """
        LangChain LLM wrapper for Enigma AI Engine.
        
        Example:
            ```python
            from enigma_engine.integrations.langchain_adapter import ForgeLLM
            from enigma_engine.core.inference import EnigmaEngine
            
            engine = EnigmaEngine(model_path="models/my_model")
            llm = ForgeLLM(engine=engine)
            
            response = llm.invoke("What is the capital of France?")
            ```
        """
        
        engine: Any = None
        model_name: str = "forge-ai"
        temperature: float = 0.7
        max_tokens: int = 2048
        top_p: float = 0.95
        top_k: int = 50
        stop_sequences: Optional[list[str]] = None
        streaming: bool = False
        
        def __init__(
            self,
            engine: Any = None,
            inference_fn: callable = None,
            **kwargs
        ):
            """
            Initialize ForgeLLM.
            
            Args:
                engine: EnigmaEngine instance
                inference_fn: Alternative inference function
                **kwargs: Additional parameters
            """
            super().__init__(**kwargs)
            self._engine = engine
            self._inference_fn = inference_fn
        
        @property
        def _llm_type(self) -> str:
            """Return identifier for the LLM."""
            return "forge-ai"
        
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Return identifying parameters."""
            return {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            }
        
        def _call(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> str:
            """
            Run the LLM on the given prompt.
            
            Args:
                prompt: The prompt to generate from
                stop: List of stop sequences
                run_manager: Callback manager
                **kwargs: Additional parameters
            """
            # Merge stop sequences
            all_stops = list(self.stop_sequences or [])
            if stop:
                all_stops.extend(stop)
            
            # Get generation parameters
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            top_p = kwargs.get("top_p", self.top_p)
            top_k = kwargs.get("top_k", self.top_k)
            
            # Generate
            if self._inference_fn:
                response = self._inference_fn(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif self._engine:
                response = self._engine.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
            else:
                raise ValueError("No engine or inference function provided")
            
            # Apply stop sequences
            for s in all_stops:
                if s in response:
                    response = response.split(s)[0]
            
            return response
        
        def _stream(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> Iterator[str]:
            """Stream the LLM output."""
            # Check if engine supports streaming
            if self._engine and hasattr(self._engine, 'generate_stream'):
                for token in self._engine.generate_stream(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature)
                ):
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                    yield token
            else:
                # Fall back to non-streaming
                response = self._call(prompt, stop, run_manager, **kwargs)
                yield response
    
    
    class ForgeChatModel(BaseChatModel):
        """
        LangChain Chat Model wrapper for Enigma AI Engine.
        
        Example:
            ```python
            from enigma_engine.integrations.langchain_adapter import ForgeChatModel
            from langchain_core.messages import HumanMessage, SystemMessage
            
            chat = ForgeChatModel(engine=engine)
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello!")
            ]
            response = chat.invoke(messages)
            ```
        """
        
        engine: Any = None
        model_name: str = "forge-chat"
        temperature: float = 0.7
        max_tokens: int = 2048
        
        def __init__(
            self,
            engine: Any = None,
            inference_fn: callable = None,
            **kwargs
        ):
            """Initialize ForgeChatModel."""
            super().__init__(**kwargs)
            self._engine = engine
            self._inference_fn = inference_fn
        
        @property
        def _llm_type(self) -> str:
            return "forge-chat"
        
        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any
        ) -> ChatResult:
            """
            Generate chat completion.
            
            Args:
                messages: List of messages
                stop: Stop sequences
                run_manager: Callback manager
            """
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Generate response
            response_text = self._generate_response(prompt, stop, **kwargs)
            
            # Create result
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
        
        def _messages_to_prompt(self, messages: list[BaseMessage]) -> str:
            """Convert messages to a prompt string."""
            parts = []
            
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    parts.append(f"System: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    parts.append(f"Assistant: {msg.content}")
                elif isinstance(msg, ChatMessage):
                    parts.append(f"{msg.role}: {msg.content}")
                else:
                    parts.append(f"User: {msg.content}")
            
            parts.append("Assistant:")
            return "\n".join(parts)
        
        def _generate_response(
            self,
            prompt: str,
            stop: Optional[list[str]] = None,
            **kwargs: Any
        ) -> str:
            """Generate response from prompt."""
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            
            if self._inference_fn:
                response = self._inference_fn(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif self._engine:
                response = self._engine.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                raise ValueError("No engine or inference function provided")
            
            # Apply stop sequences
            if stop:
                for s in stop:
                    if s in response:
                        response = response.split(s)[0]
            
            return response
    
    
    class ForgeEmbeddings(Embeddings):
        """
        LangChain Embeddings wrapper for Enigma AI Engine.
        
        Example:
            ```python
            from enigma_engine.integrations.langchain_adapter import ForgeEmbeddings
            
            embeddings = ForgeEmbeddings(embedding_fn=my_embed_fn)
            
            vectors = embeddings.embed_documents(["Hello", "World"])
            query_vector = embeddings.embed_query("Hi there")
            ```
        """
        
        def __init__(
            self,
            embedding_fn: callable = None,
            embedding_model: Any = None,
            dimension: int = 768
        ):
            """
            Initialize ForgeEmbeddings.
            
            Args:
                embedding_fn: Function to compute embeddings
                embedding_model: Embedding model instance
                dimension: Embedding dimension
            """
            self._embedding_fn = embedding_fn
            self._embedding_model = embedding_model
            self._dimension = dimension
        
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """
            Embed a list of documents.
            
            Args:
                texts: List of texts to embed
            
            Returns:
                List of embedding vectors
            """
            embeddings = []
            
            for text in texts:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            
            return embeddings
        
        def embed_query(self, text: str) -> list[float]:
            """
            Embed a single query.
            
            Args:
                text: Query text
            
            Returns:
                Embedding vector
            """
            return self._get_embedding(text)
        
        def _get_embedding(self, text: str) -> list[float]:
            """Get embedding for a single text."""
            if self._embedding_fn:
                result = self._embedding_fn(text)
                if hasattr(result, 'tolist'):
                    return result.tolist()
                return list(result)
            
            if self._embedding_model:
                if hasattr(self._embedding_model, 'encode'):
                    result = self._embedding_model.encode(text)
                    if hasattr(result, 'tolist'):
                        return result.tolist()
                    return list(result)
                if hasattr(self._embedding_model, 'embed'):
                    return self._embedding_model.embed(text)
            
            # Fallback: simple hash-based pseudo-embedding
            return self._hash_embedding(text)
        
        def _hash_embedding(self, text: str) -> list[float]:
            """Generate pseudo-embedding from text hash."""
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            embedding = []
            for i in range(self._dimension):
                byte_idx = i % len(h)
                val = (h[byte_idx] / 255.0) - 0.5
                embedding.append(val)
            return embedding


    def create_forge_llm(
        engine: Any = None,
        inference_fn: callable = None,
        **kwargs
    ) -> ForgeLLM:
        """Create a ForgeLLM instance."""
        return ForgeLLM(engine=engine, inference_fn=inference_fn, **kwargs)
    
    
    def create_forge_chat_model(
        engine: Any = None,
        inference_fn: callable = None,
        **kwargs
    ) -> ForgeChatModel:
        """Create a ForgeChatModel instance."""
        return ForgeChatModel(engine=engine, inference_fn=inference_fn, **kwargs)
    
    
    def create_forge_embeddings(
        embedding_fn: callable = None,
        **kwargs
    ) -> ForgeEmbeddings:
        """Create a ForgeEmbeddings instance."""
        return ForgeEmbeddings(embedding_fn=embedding_fn, **kwargs)

else:
    # Stubs when LangChain not available
    class ForgeLLM:
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain-core required for LangChain integration")
    
    class ForgeChatModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain-core required for LangChain integration")
    
    class ForgeEmbeddings:
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain-core required for LangChain integration")
    
    def create_forge_llm(*args, **kwargs):
        raise ImportError("langchain-core required for LangChain integration")
    
    def create_forge_chat_model(*args, **kwargs):
        raise ImportError("langchain-core required for LangChain integration")
    
    def create_forge_embeddings(*args, **kwargs):
        raise ImportError("langchain-core required for LangChain integration")
