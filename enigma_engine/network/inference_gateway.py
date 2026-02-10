"""
Inference Gateway - Unified entry point for local and remote inference

Automatically routes inference requests to the best available target:
- Local GPU (if capable)
- Local CPU (if local preferred)
- Remote server (if offloading enabled)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .failover import FailoverManager
from .load_balancer import BalancingStrategy, LoadBalancer
from .remote_offloading import (
    OffloadCriteria,
    OffloadDecision,
    get_remote_offloader,
)

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference execution mode."""
    AUTO = "auto"           # Automatically choose best target
    LOCAL_ONLY = "local"    # Always local, never offload
    REMOTE_ONLY = "remote"  # Always remote (fail if unavailable)
    PREFER_LOCAL = "prefer_local"    # Local if capable, else remote
    PREFER_REMOTE = "prefer_remote"  # Remote if available, else local


@dataclass
class InferenceResult:
    """Result from inference execution."""
    success: bool
    output: Any
    execution_time_ms: float
    executed_on: str              # "local" or server address
    model_used: Optional[str] = None
    tokens_generated: int = 0
    error: Optional[str] = None


class InferenceGateway:
    """
    Unified gateway for all inference requests.
    
    Provides a single entry point that automatically:
    - Detects hardware capabilities
    - Routes to local or remote based on task requirements
    - Handles failover when remote unavailable
    - Tracks statistics
    
    Usage:
        gateway = InferenceGateway()
        
        # Simple text generation
        result = gateway.generate("Write a poem about AI")
        
        # Chat with context
        result = gateway.chat("Hello!", history=[...])
        
        # Code generation (may route to specialized model)
        result = gateway.generate_code("Sort a list in Python")
    """
    
    def __init__(
        self,
        mode: InferenceMode = InferenceMode.AUTO,
        enable_failover: bool = True
    ):
        """
        Initialize inference gateway.
        
        Args:
            mode: Default inference mode
            enable_failover: Enable automatic failover on errors
        """
        self.mode = mode
        self.enable_failover = enable_failover
        
        # Components
        self._offloader = get_remote_offloader()
        self._load_balancer = LoadBalancer(BalancingStrategy.ADAPTIVE)
        self._failover = FailoverManager() if enable_failover else None
        
        # Local inference engine (lazy loaded)
        self._local_engine = None
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "local_requests": 0,
            "remote_requests": 0,
            "fallbacks": 0,
            "errors": 0,
        }
    
    @property
    def local_engine(self):
        """Get or create local inference engine."""
        if self._local_engine is None:
            try:
                from ..core.inference import EnigmaEngine
                self._local_engine = EnigmaEngine()
            except Exception as e:
                logger.error(f"Failed to create local engine: {e}")
        return self._local_engine
    
    def add_server(
        self,
        address: str,
        port: int,
        weight: float = 1.0,
        is_primary: bool = False
    ):
        """Add a remote server to the pool."""
        self._load_balancer.add_server(address, port, weight)
        if self._failover:
            self._failover.add_server(address, port, is_primary)
    
    def remove_server(self, address: str, port: int):
        """Remove a server from the pool."""
        self._load_balancer.remove_server(address, port)
        if self._failover:
            self._failover.remove_server(address, port)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        model_id: Optional[str] = None,
        mode: Optional[InferenceMode] = None,
        **kwargs
    ) -> InferenceResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model_id: Specific model to use (optional)
            mode: Override default inference mode
            **kwargs: Additional generation parameters
            
        Returns:
            InferenceResult with generated text
        """
        self._stats["total_requests"] += 1
        mode = mode or self.mode
        start_time = time.time()
        
        # Estimate task requirements
        criteria = OffloadCriteria(
            model_size_mb=1000,  # Estimate
            requires_gpu=True,
            estimated_time_s=max_tokens * 0.05,  # Rough estimate
            allow_remote=mode not in (InferenceMode.LOCAL_ONLY,),
            prefer_remote=mode == InferenceMode.PREFER_REMOTE,
        )
        
        # Decide where to run
        if mode == InferenceMode.LOCAL_ONLY:
            return self._generate_local(prompt, max_tokens, temperature, start_time, **kwargs)
        elif mode == InferenceMode.REMOTE_ONLY:
            return self._generate_remote(prompt, max_tokens, temperature, start_time, **kwargs)
        else:
            # Auto or prefer modes
            decision, server = self._offloader.decide(criteria)
            
            if decision == OffloadDecision.LOCAL:
                return self._generate_local(prompt, max_tokens, temperature, start_time, **kwargs)
            elif decision == OffloadDecision.REMOTE:
                result = self._generate_remote(
                    prompt, max_tokens, temperature, start_time,
                    server=server, **kwargs
                )
                # Fallback on error
                if not result.success and self.enable_failover:
                    self._stats["fallbacks"] += 1
                    return self._generate_local(prompt, max_tokens, temperature, start_time, **kwargs)
                return result
            else:
                return InferenceResult(
                    success=False,
                    output="",
                    execution_time_ms=0,
                    executed_on="none",
                    error="No inference capacity available"
                )
    
    def _generate_local(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        start_time: float,
        **kwargs
    ) -> InferenceResult:
        """Generate locally."""
        self._stats["local_requests"] += 1
        
        try:
            engine = self.local_engine
            if engine is None:
                raise RuntimeError("No local inference engine")
            
            output = engine.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return InferenceResult(
                success=True,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000,
                executed_on="local",
                model_used=getattr(engine, 'model_name', 'unknown'),
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Local generation failed: {e}")
            return InferenceResult(
                success=False,
                output="",
                execution_time_ms=(time.time() - start_time) * 1000,
                executed_on="local",
                error=str(e)
            )
    
    def _generate_remote(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        start_time: float,
        server: Optional[Any] = None,
        **kwargs
    ) -> InferenceResult:
        """Generate on remote server."""
        self._stats["remote_requests"] += 1
        
        # Get server from load balancer if not specified
        if server is None:
            server = self._load_balancer.get_server()
        
        if server is None:
            return InferenceResult(
                success=False,
                output="",
                execution_time_ms=0,
                executed_on="remote",
                error="No remote servers available"
            )
        
        try:
            from ..comms.remote_client import RemoteClient
            
            base_url = f"http://{server.address}:{server.port}"
            client = RemoteClient(base_url)
            
            self._load_balancer.mark_request_start(server)
            request_start = time.time()
            
            output = client.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            latency = (time.time() - request_start) * 1000
            self._load_balancer.mark_request_end(server, True, latency)
            
            return InferenceResult(
                success=True,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000,
                executed_on=f"{server.address}:{server.port}",
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            if server:
                self._load_balancer.mark_request_end(server, False, 0)
            logger.error(f"Remote generation failed: {e}")
            return InferenceResult(
                success=False,
                output="",
                execution_time_ms=(time.time() - start_time) * 1000,
                executed_on="remote",
                error=str(e)
            )
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None,
        mode: Optional[InferenceMode] = None,
        **kwargs
    ) -> InferenceResult:
        """
        Chat with the AI.
        
        Args:
            message: User message
            history: Conversation history
            system_prompt: System prompt
            mode: Inference mode override
            **kwargs: Additional parameters
            
        Returns:
            InferenceResult with response
        """
        # Build prompt from history
        history = history or []
        prompt = ""
        
        if system_prompt:
            prompt += f"System: {system_prompt}\n\n"
        
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        
        prompt += f"User: {message}\nAssistant:"
        
        return self.generate(prompt, mode=mode, **kwargs)
    
    def get_stats(self) -> Dict:
        """Get gateway statistics."""
        return {
            **self._stats,
            "load_balancer": self._load_balancer.get_stats(),
            "failover": self._failover.get_status() if self._failover else None,
        }
    
    def start(self):
        """Start background services."""
        if self._failover:
            self._failover.start()
    
    def stop(self):
        """Stop background services."""
        if self._failover:
            self._failover.stop()


# Global instance
_gateway: Optional[InferenceGateway] = None


def get_inference_gateway() -> InferenceGateway:
    """Get the global inference gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = InferenceGateway()
    return _gateway
