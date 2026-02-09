"""
Model Interpretability for Enigma AI Engine

Explain and interpret model predictions.

Features:
- Feature attribution
- SHAP values
- Integrated gradients
- Attention visualization
- Input importance
- Token contribution

Usage:
    from enigma_engine.core.interpretability import Explainer, explain_prediction
    
    # Create explainer
    explainer = Explainer(model, tokenizer)
    
    # Explain prediction
    explanation = explainer.explain("What is 2+2?", "4")
    
    # Get token importance
    importance = explainer.token_importance("The quick brown fox")
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


@dataclass
class TokenImportance:
    """Importance scores for tokens."""
    token: str
    token_id: int
    score: float
    normalized_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "score": self.score,
            "normalized_score": self.normalized_score
        }


@dataclass
class Explanation:
    """Explanation for a model prediction."""
    input_text: str
    output_text: str
    method: str
    token_importances: List[TokenImportance]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "method": self.method,
            "importances": [t.to_dict() for t in self.token_importances],
            "metadata": self.metadata
        }
    
    def __str__(self) -> str:
        lines = [
            f"Explanation ({self.method})",
            f"Input: {self.input_text[:100]}...",
            f"Output: {self.output_text[:100]}...",
            "",
            "Token Importances:"
        ]
        
        sorted_tokens = sorted(
            self.token_importances,
            key=lambda x: abs(x.score),
            reverse=True
        )
        
        for tok in sorted_tokens[:10]:
            bar = "+" * int(abs(tok.normalized_score) * 20)
            sign = "+" if tok.score > 0 else "-"
            lines.append(f"  {tok.token:20} {sign}{bar} ({tok.score:.4f})")
        
        return "\n".join(lines)


class GradientExplainer:
    """Gradient-based explanations."""
    
    def __init__(self, model: Any, embedding_layer: Optional[str] = None):
        """
        Initialize gradient explainer.
        
        Args:
            model: Model to explain
            embedding_layer: Name of embedding layer
        """
        self._model = model
        self._embedding_layer = embedding_layer
    
    def get_embedding_layer(self) -> Optional[Any]:
        """Find embedding layer."""
        if not HAS_TORCH:
            return None
        
        if self._embedding_layer:
            for name, module in self._model.named_modules():
                if name == self._embedding_layer:
                    return module
        
        # Auto-detect
        for name, module in self._model.named_modules():
            if isinstance(module, nn.Embedding):
                return module
            if "embed" in name.lower():
                return module
        
        return None
    
    def compute_gradients(
        self,
        input_ids: Any,
        target_ids: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Compute input gradients.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs (for computing loss)
            
        Returns:
            Gradients with respect to input
        """
        if not HAS_TORCH:
            return None
        
        embedding_layer = self.get_embedding_layer()
        if embedding_layer is None:
            logger.warning("No embedding layer found")
            return None
        
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Get embeddings
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()
        
        # Forward pass with custom embedding
        try:
            # Try to run model with pre-computed embeddings
            if hasattr(self._model, 'forward_with_embeddings'):
                outputs = self._model.forward_with_embeddings(embeddings)
            else:
                # Standard forward
                outputs = self._model(input_ids)
            
            # Get logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss
            if target_ids is not None:
                if not isinstance(target_ids, torch.Tensor):
                    target_ids = torch.tensor([target_ids])
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            else:
                # Use max logit as target
                loss = logits.max()
            
            # Backward pass
            loss.backward()
            
            return embeddings.grad
            
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return None
    
    def saliency_scores(
        self,
        input_ids: Any,
        target_ids: Optional[Any] = None
    ) -> List[float]:
        """
        Compute saliency scores for input tokens.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs
            
        Returns:
            Saliency score per token
        """
        grads = self.compute_gradients(input_ids, target_ids)
        
        if grads is None:
            return []
        
        # L2 norm per token
        scores = grads.norm(dim=-1).squeeze().tolist()
        
        if isinstance(scores, float):
            scores = [scores]
        
        return scores


class IntegratedGradients:
    """Integrated gradients attribution."""
    
    def __init__(
        self,
        model: Any,
        embedding_layer: Optional[str] = None,
        steps: int = 50
    ):
        """
        Initialize integrated gradients.
        
        Args:
            model: Model to explain
            embedding_layer: Name of embedding layer
            steps: Integration steps
        """
        self._model = model
        self._embedding_layer = embedding_layer
        self._steps = steps
        self._grad_explainer = GradientExplainer(model, embedding_layer)
    
    def attribute(
        self,
        input_ids: Any,
        basline_ids: Optional[Any] = None,
        target_ids: Optional[Any] = None
    ) -> List[float]:
        """
        Compute integrated gradients attribution.
        
        Args:
            input_ids: Input token IDs
            baseline_ids: Baseline (default: padding tokens)
            target_ids: Target output
            
        Returns:
            Attribution scores
        """
        if not HAS_TORCH:
            return []
        
        embedding_layer = self._grad_explainer.get_embedding_layer()
        if embedding_layer is None:
            return []
        
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Baseline (padding tokens or zeros)
        if basline_ids is None:
            baseline = torch.zeros_like(embedding_layer(input_ids))
        else:
            if not isinstance(basline_ids, torch.Tensor):
                basline_ids = torch.tensor([basline_ids])
            baseline = embedding_layer(basline_ids)
        
        # Reference embedding
        embeddings = embedding_layer(input_ids)
        
        # Accumulate gradients
        integrated_grads = torch.zeros_like(embeddings)
        
        for step in range(self._steps):
            alpha = step / self._steps
            
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (embeddings - baseline)
            interpolated.requires_grad_(True)
            
            try:
                # Forward pass
                if hasattr(self._model, 'forward_with_embeddings'):
                    outputs = self._model.forward_with_embeddings(interpolated)
                else:
                    outputs = self._model(input_ids)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Loss
                if target_ids is not None:
                    loss = logits[:, :, target_ids].sum()
                else:
                    loss = logits.max()
                
                loss.backward()
                
                if interpolated.grad is not None:
                    integrated_grads += interpolated.grad
                
            except Exception as e:
                logger.debug(f"Step {step} failed: {e}")
                continue
        
        # Scale and compute attribution
        integrated_grads = integrated_grads / self._steps
        attribution = (embeddings - baseline) * integrated_grads
        
        # Sum over embedding dimension
        scores = attribution.sum(dim=-1).squeeze().tolist()
        
        if isinstance(scores, float):
            scores = [scores]
        
        return scores


class AttentionExplainer:
    """Attention-based explanations."""
    
    def __init__(self, model: Any):
        """Initialize attention explainer."""
        self._model = model
    
    def get_attention_weights(
        self,
        input_ids: Any,
        layer: int = -1
    ) -> Optional[Any]:
        """
        Get attention weights.
        
        Args:
            input_ids: Input token IDs
            layer: Layer index (-1 for last)
            
        Returns:
            Attention weights tensor
        """
        if not HAS_TORCH:
            return None
        
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Collect attention weights
        attention_weights = []
        
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1:
                attention_weights.append(out[1])
        
        # Find attention layers and attach hooks
        hooks = []
        for name, module in self._model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(hook)
                hooks.append(handle)
        
        try:
            with torch.no_grad():
                self._model(input_ids)
        finally:
            for h in hooks:
                h.remove()
        
        if attention_weights:
            if layer < 0:
                layer = len(attention_weights) + layer
            if 0 <= layer < len(attention_weights):
                return attention_weights[layer]
        
        return None
    
    def attention_rollout(
        self,
        input_ids: Any,
        head_fusion: str = "mean"
    ) -> List[float]:
        """
        Compute attention rollout scores.
        
        Rolls attention through layers to compute token importance.
        
        Args:
            input_ids: Input token IDs
            head_fusion: How to combine heads (mean, max)
            
        Returns:
            Importance score per token
        """
        if not HAS_TORCH:
            return []
        
        # Collect all attention matrices
        all_attentions = []
        
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1:
                all_attentions.append(out[1])
        
        hooks = []
        for name, module in self._model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(hook)
                hooks.append(handle)
        
        # Ensure tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        try:
            with torch.no_grad():
                self._model(input_ids)
        finally:
            for h in hooks:
                h.remove()
        
        if not all_attentions:
            return []
        
        # Process attention matrices
        seq_len = input_ids.size(1)
        rollout = torch.eye(seq_len)
        
        for attention in all_attentions:
            # Fuse heads
            if attention.dim() == 4:  # (batch, heads, seq, seq)
                if head_fusion == "mean":
                    attention = attention.mean(dim=1)
                else:
                    attention = attention.max(dim=1)[0]
            
            attention = attention[0]  # Remove batch dimension
            
            # Add identity and normalize
            attention = attention + torch.eye(attention.size(0))
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Roll through
            rollout = torch.matmul(attention, rollout)
        
        # Get importance from CLS token (first) or average
        importance = rollout[0].tolist()
        
        return importance


class Explainer:
    """
    High-level explainer combining multiple methods.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        methods: Optional[List[str]] = None
    ):
        """
        Initialize explainer.
        
        Args:
            model: Model to explain
            tokenizer: Tokenizer
            methods: Explanation methods to use
        """
        self._model = model
        self._tokenizer = tokenizer
        self._methods = methods or ["gradient", "attention"]
        
        # Initialize sub-explainers
        self._gradient = GradientExplainer(model)
        self._integrated = IntegratedGradients(model)
        self._attention = AttentionExplainer(model)
    
    def explain(
        self,
        input_text: str,
        output_text: str = "",
        method: str = "gradient"
    ) -> Explanation:
        """
        Explain a prediction.
        
        Args:
            input_text: Input text
            output_text: Generated output
            method: Explanation method
            
        Returns:
            Explanation object
        """
        # Tokenize
        if hasattr(self._tokenizer, 'encode'):
            input_ids = self._tokenizer.encode(input_text)
        else:
            input_ids = list(input_text.encode())
        
        target_ids = None
        if output_text and hasattr(self._tokenizer, 'encode'):
            target_ids = self._tokenizer.encode(output_text)
        
        # Get scores based on method
        if method == "gradient":
            scores = self._gradient.saliency_scores(input_ids, target_ids)
        elif method == "integrated":
            scores = self._integrated.attribute(input_ids, target_ids=target_ids)
        elif method == "attention":
            scores = self._attention.attention_rollout(input_ids)
        else:
            scores = self._gradient.saliency_scores(input_ids, target_ids)
        
        # Create token importances
        token_importances = []
        
        if hasattr(self._tokenizer, 'decode'):
            for i, (token_id, score) in enumerate(zip(input_ids, scores)):
                token = self._tokenizer.decode([token_id])
                token_importances.append(TokenImportance(
                    token=token,
                    token_id=token_id,
                    score=score
                ))
        else:
            for i, score in enumerate(scores):
                token_importances.append(TokenImportance(
                    token=f"[{i}]",
                    token_id=i,
                    score=score
                ))
        
        # Normalize scores
        if token_importances:
            max_score = max(abs(t.score) for t in token_importances) or 1
            for t in token_importances:
                t.normalized_score = t.score / max_score
        
        return Explanation(
            input_text=input_text,
            output_text=output_text,
            method=method,
            token_importances=token_importances
        )
    
    def token_importance(
        self,
        text: str,
        method: str = "gradient"
    ) -> List[TokenImportance]:
        """
        Get token importance scores.
        
        Args:
            text: Input text
            method: Explanation method
            
        Returns:
            List of token importances
        """
        explanation = self.explain(text, method=method)
        return explanation.token_importances
    
    def compare_methods(
        self,
        input_text: str,
        output_text: str = ""
    ) -> Dict[str, Explanation]:
        """
        Compare multiple explanation methods.
        
        Args:
            input_text: Input text
            output_text: Generated output
            
        Returns:
            Dictionary of method -> explanation
        """
        results = {}
        
        for method in ["gradient", "integrated", "attention"]:
            try:
                results[method] = self.explain(input_text, output_text, method)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
        
        return results


# Convenience function
def explain_prediction(
    model: Any,
    tokenizer: Any,
    input_text: str,
    output_text: str = "",
    method: str = "gradient"
) -> Explanation:
    """
    Explain a model prediction.
    
    Args:
        model: Model to explain
        tokenizer: Tokenizer
        input_text: Input text
        output_text: Generated output
        method: Explanation method
        
    Returns:
        Explanation object
    """
    explainer = Explainer(model, tokenizer)
    return explainer.explain(input_text, output_text, method)
