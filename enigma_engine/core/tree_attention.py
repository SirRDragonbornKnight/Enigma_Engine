"""
Tree Attention

Parallel token evaluation using tree-based attention structures.
Enables efficient speculative decoding and parallel hypothesis exploration.

FILE: enigma_engine/core/tree_attention.py
TYPE: Core/Inference
MAIN CLASSES: TreeAttention, SpeculativeDecoder, ParallelBeamSearch
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeStrategy(Enum):
    """Tree attention strategies."""
    FLAT = "flat"           # Standard attention
    BINARY = "binary"       # Binary tree structure
    TOP_K = "top_k"         # Top-K branches
    BEAM = "beam"           # Beam search tree
    SPECULATIVE = "speculative"  # Speculative decoding tree


@dataclass
class TreeNode:
    """Node in the attention tree."""
    token_id: int
    log_prob: float
    parent: Optional['TreeNode'] = None
    children: list['TreeNode'] = field(default_factory=list)
    depth: int = 0
    cumulative_prob: float = 0.0
    hidden_state: Optional[Any] = None
    kv_cache: Optional[Any] = None
    is_accepted: bool = False
    
    def __lt__(self, other):
        """For heap operations."""
        return self.cumulative_prob > other.cumulative_prob  # Max heap


@dataclass
class TreeConfig:
    """Tree attention configuration."""
    max_depth: int = 5
    max_branches: int = 4
    min_prob_threshold: float = 0.01
    strategy: TreeStrategy = TreeStrategy.TOP_K
    draft_model_lookahead: int = 4
    accept_threshold: float = 0.9


if HAS_TORCH:
    
    class TreeMask:
        """
        Generate attention masks for tree structures.
        """
        
        @staticmethod
        def create_tree_mask(
            tree_structure: list[list[int]],
            device: torch.device = None
        ) -> torch.Tensor:
            """
            Create causal attention mask for tree structure.
            
            Args:
                tree_structure: List of [parent_idx] for each position
                device: Torch device
            
            Returns:
                Attention mask tensor
            """
            seq_len = len(tree_structure)
            mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            
            for i, parents in enumerate(tree_structure):
                # Each position can attend to itself
                mask[i, i] = True
                
                # And to all ancestors
                for parent_idx in parents:
                    if parent_idx >= 0:
                        mask[i, parent_idx] = True
            
            return mask
        
        @staticmethod
        def expand_tree_mask(
            mask: torch.Tensor,
            num_heads: int
        ) -> torch.Tensor:
            """
            Expand mask for multi-head attention.
            
            Args:
                mask: [seq_len, seq_len] mask
                num_heads: Number of attention heads
            
            Returns:
                [num_heads, seq_len, seq_len] expanded mask
            """
            return mask.unsqueeze(0).expand(num_heads, -1, -1)
    
    
    class TreeAttention(nn.Module):
        """
        Tree-structured attention for parallel token evaluation.
        
        Instead of processing tokens sequentially, this evaluates
        multiple token paths simultaneously in a tree structure.
        """
        
        def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            config: TreeConfig = None
        ) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.config = config or TreeConfig()
            
            # Projections
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)
            
            self._current_tree: list[TreeNode] = []
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            tree_mask: Optional[torch.Tensor] = None,
            past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            use_tree: bool = True
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass with tree attention.
            
            Args:
                hidden_states: [batch, seq, hidden]
                tree_mask: Tree attention mask
                past_kv: Cached key/value tensors
                use_tree: Whether to use tree structure
            
            Returns:
                Tuple of (output, (key_cache, value_cache))
            """
            batch_size, seq_len, _ = hidden_states.shape
            
            # Project queries, keys, values
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Concatenate with past KV if provided
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Apply tree mask if provided
            if tree_mask is not None:
                # Expand mask for batch and heads
                if tree_mask.dim() == 2:
                    tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~tree_mask, float('-inf'))
            else:
                # Standard causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.size(2), dtype=torch.bool, device=scores.device),
                    diagonal=k.size(2) - seq_len + 1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))
            
            # Softmax and attention
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            output = self.o_proj(output)
            
            return output, (k, v)
        
        def build_tree(
            self,
            logits: torch.Tensor,
            root_state: torch.Tensor
        ) -> list[TreeNode]:
            """
            Build attention tree from logits.
            
            Args:
                logits: [vocab_size] logits for next token
                root_state: Hidden state at root
            
            Returns:
                List of tree nodes
            """
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k tokens
            top_k = min(self.config.max_branches, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Create root node
            root = TreeNode(
                token_id=-1,  # Root
                log_prob=0.0,
                depth=0,
                cumulative_prob=1.0,
                hidden_state=root_state
            )
            
            nodes = [root]
            
            # Build tree with BFS
            queue = [(root, 0)]
            
            while queue and len(nodes) < self.config.max_branches * self.config.max_depth:
                parent, depth = queue.pop(0)
                
                if depth >= self.config.max_depth:
                    continue
                
                prob_idx = min(top_probs.size(-1) - 1, depth)
                
                for i in range(min(self.config.max_branches, top_k)):
                    token_prob = top_probs[i].item()
                    
                    if token_prob < self.config.min_prob_threshold:
                        continue
                    
                    child = TreeNode(
                        token_id=top_indices[i].item(),
                        log_prob=math.log(token_prob + 1e-10),
                        parent=parent,
                        depth=depth + 1,
                        cumulative_prob=parent.cumulative_prob * token_prob
                    )
                    
                    parent.children.append(child)
                    nodes.append(child)
                    queue.append((child, depth + 1))
            
            self._current_tree = nodes
            return nodes
    
    
    class SpeculativeDecoder:
        """
        Speculative decoding using draft model and tree verification.
        
        Uses a smaller draft model to generate candidate tokens,
        then verifies them in parallel with the main model.
        """
        
        def __init__(
            self,
            main_model: nn.Module,
            draft_model: nn.Module,
            config: TreeConfig = None
        ) -> None:
            self.main_model = main_model
            self.draft_model = draft_model
            self.config = config or TreeConfig()
            self._stats = {
                "total_tokens": 0,
                "accepted_tokens": 0,
                "draft_calls": 0,
                "verify_calls": 0
            }
        
        @torch.no_grad()
        def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 50,
            **kwargs
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            """
            Generate tokens using speculative decoding.
            
            Args:
                input_ids: [batch, seq] input token IDs
                max_new_tokens: Maximum tokens to generate
            
            Returns:
                Tuple of (output_ids, statistics)
            """
            generated = input_ids.clone()
            lookahead = self.config.draft_model_lookahead
            
            tokens_generated = 0
            
            while tokens_generated < max_new_tokens:
                # 1. Generate draft tokens
                draft_tokens, draft_probs = self._generate_draft(
                    generated, 
                    num_tokens=lookahead
                )
                self._stats["draft_calls"] += 1
                
                # 2. Verify with main model
                accepted, next_token = self._verify_tokens(
                    generated,
                    draft_tokens,
                    draft_probs
                )
                self._stats["verify_calls"] += 1
                
                # 3. Accept verified tokens
                num_accepted = accepted.sum().item()
                self._stats["accepted_tokens"] += num_accepted
                self._stats["total_tokens"] += lookahead
                
                if num_accepted > 0:
                    accepted_tokens = draft_tokens[:, :num_accepted]
                    generated = torch.cat([generated, accepted_tokens], dim=1)
                    tokens_generated += num_accepted
                
                # Always append at least one token
                if next_token is not None:
                    generated = torch.cat([generated, next_token], dim=1)
                    tokens_generated += 1
                
                # Check for EOS
                if hasattr(self.main_model, 'config'):
                    eos_id = getattr(self.main_model.config, 'eos_token_id', None)
                    if eos_id and generated[0, -1].item() == eos_id:
                        break
            
            return generated, self._stats.copy()
        
        def _generate_draft(
            self,
            input_ids: torch.Tensor,
            num_tokens: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Generate draft tokens with draft model."""
            draft_tokens = []
            draft_probs = []
            
            current = input_ids
            
            for _ in range(num_tokens):
                outputs = self.draft_model(current)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits[:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                draft_tokens.append(next_token)
                draft_probs.append(probs.gather(-1, next_token))
                
                current = torch.cat([current, next_token], dim=1)
            
            return torch.cat(draft_tokens, dim=1), torch.cat(draft_probs, dim=1)
        
        def _verify_tokens(
            self,
            prefix: torch.Tensor,
            draft_tokens: torch.Tensor,
            draft_probs: torch.Tensor
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Verify draft tokens with main model."""
            # Concatenate for parallel verification
            full_input = torch.cat([prefix, draft_tokens], dim=1)
            
            # Get main model logits for all positions
            outputs = self.main_model(full_input)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Extract logits for verification positions
            num_draft = draft_tokens.size(1)
            verify_start = prefix.size(1) - 1
            verify_logits = logits[:, verify_start:verify_start + num_draft, :]
            
            # Compute main model probs
            main_probs = F.softmax(verify_logits, dim=-1)
            main_token_probs = main_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Accept tokens where main prob >= draft prob
            accept_ratio = main_token_probs / (draft_probs + 1e-10)
            accepted = (accept_ratio >= self.config.accept_threshold) | (torch.rand_like(accept_ratio) < accept_ratio)
            
            # Find first rejection
            first_reject = (~accepted).float().argmax(dim=1)
            
            # Create acceptance mask
            batch_size = accepted.size(0)
            accept_mask = torch.arange(num_draft, device=accepted.device).unsqueeze(0) < first_reject.unsqueeze(1)
            
            # Sample next token from main model
            next_logits = logits[:, verify_start + first_reject[0].item(), :]
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            
            return accept_mask[0], next_token
        
        def get_acceptance_rate(self) -> float:
            """Get token acceptance rate."""
            if self._stats["total_tokens"] == 0:
                return 0.0
            return self._stats["accepted_tokens"] / self._stats["total_tokens"]
        
        def reset_stats(self) -> None:
            """Reset statistics."""
            self._stats = {
                "total_tokens": 0,
                "accepted_tokens": 0,
                "draft_calls": 0,
                "verify_calls": 0
            }
    
    
    class ParallelBeamSearch:
        """
        Beam search with parallel evaluation using tree attention.
        """
        
        def __init__(
            self,
            model: nn.Module,
            beam_width: int = 4,
            max_length: int = 50,
            length_penalty: float = 1.0
        ) -> None:
            self.model = model
            self.beam_width = beam_width
            self.max_length = max_length
            self.length_penalty = length_penalty
        
        @torch.no_grad()
        def search(
            self,
            input_ids: torch.Tensor,
            **kwargs
        ) -> list[tuple[torch.Tensor, float]]:
            """
            Perform beam search with tree attention.
            
            Args:
                input_ids: [1, seq] input token IDs
            
            Returns:
                List of (sequence, score) tuples
            """
            device = input_ids.device
            
            # Initialize beams
            beams: list[TreeNode] = []
            
            # Get initial logits
            outputs = self.model(input_ids)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs[:, -1, :]
            
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Initialize top-k beams
            top_probs, top_indices = torch.topk(log_probs[0], self.beam_width)
            
            for i in range(self.beam_width):
                node = TreeNode(
                    token_id=top_indices[i].item(),
                    log_prob=top_probs[i].item(),
                    depth=1,
                    cumulative_prob=top_probs[i].item()
                )
                beams.append(node)
            
            # Expand beams
            completed = []
            
            for step in range(self.max_length - 1):
                if not beams:
                    break
                
                # Build batch for parallel evaluation
                beam_inputs = []
                for beam in beams:
                    # Reconstruct sequence from beam
                    tokens = self._get_beam_tokens(beam)
                    beam_seq = torch.cat([
                        input_ids,
                        torch.tensor([tokens], device=device)
                    ], dim=1)
                    beam_inputs.append(beam_seq)
                
                # Pad to same length
                max_len = max(b.size(1) for b in beam_inputs)
                padded = torch.zeros(len(beam_inputs), max_len, dtype=torch.long, device=device)
                for i, b in enumerate(beam_inputs):
                    padded[i, :b.size(1)] = b
                
                # Forward pass
                outputs = self.model(padded)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get next token logits
                all_candidates = []
                for i, beam in enumerate(beams):
                    seq_len = len(self._get_beam_tokens(beam)) + input_ids.size(1)
                    beam_logits = logits[i, seq_len - 1, :]
                    log_probs = F.log_softmax(beam_logits, dim=-1)
                    
                    top_probs, top_indices = torch.topk(log_probs, self.beam_width)
                    
                    for j in range(self.beam_width):
                        new_score = beam.cumulative_prob + top_probs[j].item()
                        # Length penalty
                        normalized_score = new_score / ((beam.depth + 1) ** self.length_penalty)
                        
                        child = TreeNode(
                            token_id=top_indices[j].item(),
                            log_prob=top_probs[j].item(),
                            parent=beam,
                            depth=beam.depth + 1,
                            cumulative_prob=new_score
                        )
                        all_candidates.append((normalized_score, child))
                
                # Select top beams
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = []
                
                for score, candidate in all_candidates[:self.beam_width]:
                    # Check for EOS
                    eos_id = getattr(self.model, 'eos_token_id', 2)
                    if candidate.token_id == eos_id:
                        completed.append((candidate, score))
                    else:
                        beams.append(candidate)
                
                # Early stopping if enough completed
                if len(completed) >= self.beam_width:
                    break
            
            # Add remaining beams to completed
            for beam in beams:
                score = beam.cumulative_prob / (beam.depth ** self.length_penalty)
                completed.append((beam, score))
            
            # Sort by score and return sequences
            completed.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for beam, score in completed[:self.beam_width]:
                tokens = self._get_beam_tokens(beam)
                seq = torch.cat([
                    input_ids,
                    torch.tensor([tokens], device=device)
                ], dim=1)
                results.append((seq, score))
            
            return results
        
        def _get_beam_tokens(self, node: TreeNode) -> list[int]:
            """Get all tokens in beam from root to node."""
            tokens = []
            current = node
            while current is not None and current.token_id >= 0:
                tokens.append(current.token_id)
                current = current.parent
            return list(reversed(tokens))
    
    
    def create_tree_attention(
        hidden_size: int = 768,
        num_heads: int = 12,
        **config_kwargs
    ) -> TreeAttention:
        """
        Create tree attention module.
        
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            **config_kwargs: TreeConfig parameters
        
        Returns:
            TreeAttention module
        """
        config = TreeConfig(**config_kwargs)
        return TreeAttention(hidden_size, num_heads, config)

else:
    class TreeNode:
        pass
    
    class TreeMask:
        pass
    
    class TreeAttention:
        pass
    
    class SpeculativeDecoder:
        pass
    
    class ParallelBeamSearch:
        pass
    
    def create_tree_attention(*args, **kwargs):
        raise ImportError("PyTorch required")
