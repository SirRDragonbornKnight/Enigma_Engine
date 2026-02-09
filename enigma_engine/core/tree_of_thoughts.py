"""
Tree-of-Thoughts Reasoning

Implements Tree-of-Thoughts (ToT) prompting strategy.
Explores multiple reasoning paths and selects the best ones.

FILE: enigma_engine/core/tree_of_thoughts.py
TYPE: Advanced Reasoning
MAIN CLASSES: ThoughtTree, ThoughtNode, ToTReasoner
PAPER: "Tree of Thoughts: Deliberate Problem Solving with LLMs" (Yao et al. 2023)
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Tree search strategies."""
    BFS = "bfs"              # Breadth-first search
    DFS = "dfs"              # Depth-first search
    BEAM = "beam"            # Beam search
    BEST_FIRST = "best_first"  # Best-first search


class NodeState(Enum):
    """State of a thought node."""
    PENDING = "pending"       # Not yet explored
    EXPLORING = "exploring"   # Currently being expanded
    EVALUATED = "evaluated"   # Has been scored
    PRUNED = "pruned"         # Discarded
    SELECTED = "selected"     # On the winning path


@dataclass
class ThoughtNode:
    """A node in the thought tree."""
    id: str
    thought: str                    # The reasoning step/thought
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    score: float = 0.0              # Evaluation score
    state: NodeState = NodeState.PENDING
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        # For heap comparison
        return self.score > other.score  # Higher score = higher priority


@dataclass
class ThoughtTree:
    """Tree structure for thought exploration."""
    root_id: str
    nodes: dict[str, ThoughtNode] = field(default_factory=dict)
    problem: str = ""
    max_depth: int = 5
    branching_factor: int = 3
    created_at: float = field(default_factory=time.time)
    
    def add_node(self, node: ThoughtNode) -> str:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.id)
        return node.id
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_root(self) -> Optional[ThoughtNode]:
        """Get the root node."""
        return self.nodes.get(self.root_id)
    
    def get_children(self, node_id: str) -> list[ThoughtNode]:
        """Get children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_path_to_node(self, node_id: str) -> list[ThoughtNode]:
        """Get the path from root to a node."""
        path = []
        current_id = node_id
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))
    
    def get_best_path(self) -> list[ThoughtNode]:
        """Get the best path (highest cumulative score)."""
        # Find all leaf nodes
        leaves = [n for n in self.nodes.values() 
                  if not n.children_ids and n.state == NodeState.EVALUATED]
        if not leaves:
            return []
        
        # Find best leaf by path score
        best_leaf = max(leaves, key=lambda n: self._path_score(n.id))
        return self.get_path_to_node(best_leaf.id)
    
    def _path_score(self, node_id: str) -> float:
        """Calculate cumulative score of path to node."""
        path = self.get_path_to_node(node_id)
        return sum(n.score for n in path) / len(path) if path else 0
    
    def to_dict(self) -> dict:
        """Convert tree to dictionary."""
        return {
            "root_id": self.root_id,
            "problem": self.problem,
            "nodes": {nid: {
                "id": n.id,
                "thought": n.thought,
                "parent_id": n.parent_id,
                "children_ids": n.children_ids,
                "score": n.score,
                "state": n.state.value,
                "depth": n.depth
            } for nid, n in self.nodes.items()}
        }


class ToTReasoner:
    """Tree-of-Thoughts reasoner."""
    
    def __init__(self,
                 generator: Callable[[str, int], list[str]] = None,
                 evaluator: Callable[[str, str], float] = None,
                 strategy: SearchStrategy = SearchStrategy.BEAM,
                 max_depth: int = 5,
                 branching_factor: int = 3,
                 beam_width: int = 3):
        """
        Initialize ToT reasoner.
        
        Args:
            generator: Function to generate thoughts. Takes (context, n) and returns n thoughts.
            evaluator: Function to evaluate a thought. Takes (problem, thought) and returns score.
            strategy: Search strategy
            max_depth: Maximum tree depth
            branching_factor: Number of thoughts to generate per node
            beam_width: Beam width for beam search
        """
        self._generator = generator or self._default_generator
        self._evaluator = evaluator or self._default_evaluator
        self._strategy = strategy
        self._max_depth = max_depth
        self._branching_factor = branching_factor
        self._beam_width = beam_width
        self._node_counter = 0
        self._lock = threading.Lock()
        
    def _default_generator(self, context: str, n: int) -> list[str]:
        """Default thought generator (placeholder)."""
        return [f"Thought {i+1}: Continue reasoning..." for i in range(n)]
    
    def _default_evaluator(self, problem: str, thought: str) -> float:
        """Default thought evaluator (placeholder)."""
        # Simple heuristic: longer, more specific thoughts score higher
        score = min(len(thought) / 200, 1.0) * 0.5
        # Bonus for reasoning keywords
        reasoning_keywords = ["because", "therefore", "if", "then", "since", "thus"]
        for keyword in reasoning_keywords:
            if keyword in thought.lower():
                score += 0.1
        return min(score, 1.0)
    
    def _generate_id(self) -> str:
        """Generate unique node ID."""
        with self._lock:
            self._node_counter += 1
            return f"node_{self._node_counter}"
    
    def reason(self, problem: str) -> tuple[str, ThoughtTree]:
        """
        Perform Tree-of-Thoughts reasoning.
        
        Args:
            problem: The problem to solve
            
        Returns:
            Tuple of (final answer, thought tree)
        """
        # Initialize tree
        root_id = self._generate_id()
        root = ThoughtNode(
            id=root_id,
            thought=f"Problem: {problem}",
            state=NodeState.EVALUATED,
            score=0.5
        )
        
        tree = ThoughtTree(
            root_id=root_id,
            problem=problem,
            max_depth=self._max_depth,
            branching_factor=self._branching_factor
        )
        tree.add_node(root)
        
        # Execute search strategy
        if self._strategy == SearchStrategy.BFS:
            self._bfs_search(tree)
        elif self._strategy == SearchStrategy.DFS:
            self._dfs_search(tree)
        elif self._strategy == SearchStrategy.BEAM:
            self._beam_search(tree)
        elif self._strategy == SearchStrategy.BEST_FIRST:
            self._best_first_search(tree)
        
        # Extract best path
        best_path = tree.get_best_path()
        
        # Mark selected path
        for node in best_path:
            node.state = NodeState.SELECTED
        
        # Compose final answer
        if best_path:
            final_answer = self._compose_answer(best_path)
        else:
            final_answer = "Unable to find a solution path."
            
        return final_answer, tree
    
    def _expand_node(self, tree: ThoughtTree, node: ThoughtNode) -> list[ThoughtNode]:
        """Expand a node by generating child thoughts."""
        if node.depth >= tree.max_depth:
            return []
        
        node.state = NodeState.EXPLORING
        
        # Build context from path to this node
        path = tree.get_path_to_node(node.id)
        context = "\n".join([n.thought for n in path])
        
        # Generate thoughts
        thoughts = self._generator(context, tree.branching_factor)
        
        # Create child nodes
        children = []
        for thought in thoughts:
            child_id = self._generate_id()
            child = ThoughtNode(
                id=child_id,
                thought=thought,
                parent_id=node.id,
                depth=node.depth + 1,
                state=NodeState.PENDING
            )
            tree.add_node(child)
            children.append(child)
        
        node.state = NodeState.EVALUATED
        return children
    
    def _evaluate_node(self, tree: ThoughtTree, node: ThoughtNode):
        """Evaluate a thought node."""
        if node.state == NodeState.EVALUATED:
            return
        
        # Build full thought context
        path = tree.get_path_to_node(node.id)
        full_thought = "\n".join([n.thought for n in path])
        
        # Get evaluation score
        node.score = self._evaluator(tree.problem, full_thought)
        node.state = NodeState.EVALUATED
        
    def _bfs_search(self, tree: ThoughtTree):
        """Breadth-first search."""
        queue = [tree.root_id]
        
        while queue:
            node_id = queue.pop(0)
            node = tree.get_node(node_id)
            if not node or node.depth >= tree.max_depth:
                continue
            
            # Expand node
            children = self._expand_node(tree, node)
            
            # Evaluate and enqueue children
            for child in children:
                self._evaluate_node(tree, child)
                if child.score > 0.3:  # Pruning threshold
                    queue.append(child.id)
                else:
                    child.state = NodeState.PRUNED
    
    def _dfs_search(self, tree: ThoughtTree):
        """Depth-first search."""
        stack = [tree.root_id]
        
        while stack:
            node_id = stack.pop()
            node = tree.get_node(node_id)
            if not node or node.depth >= tree.max_depth:
                continue
            
            # Expand node
            children = self._expand_node(tree, node)
            
            # Evaluate and push children (reverse to maintain order)
            for child in reversed(children):
                self._evaluate_node(tree, child)
                if child.score > 0.3:
                    stack.append(child.id)
                else:
                    child.state = NodeState.PRUNED
    
    def _beam_search(self, tree: ThoughtTree):
        """Beam search - keep top k nodes at each level."""
        beam = [tree.root_id]
        
        for depth in range(tree.max_depth):
            candidates = []
            
            # Expand all nodes in beam
            for node_id in beam:
                node = tree.get_node(node_id)
                if not node:
                    continue
                children = self._expand_node(tree, node)
                
                # Evaluate children
                for child in children:
                    self._evaluate_node(tree, child)
                    candidates.append(child)
            
            if not candidates:
                break
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda n: n.score, reverse=True)
            beam = [c.id for c in candidates[:self._beam_width]]
            
            # Mark pruned
            for candidate in candidates[self._beam_width:]:
                candidate.state = NodeState.PRUNED
    
    def _best_first_search(self, tree: ThoughtTree):
        """Best-first search using priority queue."""
        # Priority queue (min-heap, so negate scores)
        pq = []
        heapq.heappush(pq, tree.get_node(tree.root_id))
        
        explored = 0
        max_explored = self._beam_width * tree.max_depth * 2  # Limit exploration
        
        while pq and explored < max_explored:
            node = heapq.heappop(pq)
            
            if node.depth >= tree.max_depth:
                continue
            
            # Expand node
            children = self._expand_node(tree, node)
            explored += 1
            
            # Evaluate and add to queue
            for child in children:
                self._evaluate_node(tree, child)
                if child.score > 0.3:
                    heapq.heappush(pq, child)
                else:
                    child.state = NodeState.PRUNED
    
    def _compose_answer(self, path: list[ThoughtNode]) -> str:
        """Compose final answer from best path."""
        # Skip root node (just contains problem)
        reasoning_steps = [node.thought for node in path[1:]]
        
        if not reasoning_steps:
            return "No reasoning path found."
        
        answer = "Reasoning path:\n"
        for i, step in enumerate(reasoning_steps, 1):
            answer += f"{i}. {step}\n"
        
        # The last step is typically the conclusion
        answer += f"\nConclusion: {reasoning_steps[-1]}"
        
        return answer


# Factory function
def create_tot_reasoner(generator: Callable = None,
                       evaluator: Callable = None,
                       strategy: str = "beam",
                       **kwargs) -> ToTReasoner:
    """
    Create a ToT reasoner.
    
    Args:
        generator: Thought generator function
        evaluator: Thought evaluator function
        strategy: Search strategy ("bfs", "dfs", "beam", "best_first")
        **kwargs: Additional arguments for ToTReasoner
        
    Returns:
        Configured ToTReasoner
    """
    strategy_map = {
        "bfs": SearchStrategy.BFS,
        "dfs": SearchStrategy.DFS,
        "beam": SearchStrategy.BEAM,
        "best_first": SearchStrategy.BEST_FIRST
    }
    
    return ToTReasoner(
        generator=generator,
        evaluator=evaluator,
        strategy=strategy_map.get(strategy, SearchStrategy.BEAM),
        **kwargs
    )


# Integration with Enigma AI Engine inference
class ForgeToTIntegration:
    """Integration layer for using ToT with Enigma AI Engine inference."""
    
    def __init__(self, engine=None):
        """
        Initialize integration.
        
        Args:
            engine: EnigmaEngine instance for inference
        """
        self._engine = engine
        self._reasoner: Optional[ToTReasoner] = None
        
    def set_engine(self, engine):
        """Set the inference engine."""
        self._engine = engine
        self._reasoner = None  # Reset reasoner
        
    def _create_generator(self):
        """Create thought generator using Enigma AI Engine."""
        def generator(context: str, n: int) -> list[str]:
            if not self._engine:
                return [f"Thought {i}: [No engine available]" for i in range(n)]
            
            thoughts = []
            prompt = f"Given the following reasoning context, generate the next step:\n\n{context}\n\nNext step:"
            
            for _ in range(n):
                try:
                    response = self._engine.generate(
                        prompt,
                        max_tokens=100,
                        temperature=0.8  # Higher for diversity
                    )
                    thoughts.append(response.strip())
                except Exception as e:
                    logger.error(f"Generator error: {e}")
                    thoughts.append("Continue reasoning...")
            
            return thoughts
        
        return generator
    
    def _create_evaluator(self):
        """Create thought evaluator using Enigma AI Engine."""
        def evaluator(problem: str, thought: str) -> float:
            if not self._engine:
                # Fallback to simple heuristic
                return min(len(thought) / 200, 1.0) * 0.5
            
            prompt = f"""Rate the quality of this reasoning (0-10):
Problem: {problem}
Reasoning: {thought}
Score (0-10):"""
            
            try:
                response = self._engine.generate(
                    prompt,
                    max_tokens=5,
                    temperature=0.1
                )
                # Extract number
                import re
                match = re.search(r'\d+', response)
                if match:
                    score = int(match.group()) / 10.0
                    return min(max(score, 0.0), 1.0)
            except Exception as e:
                logger.error(f"Evaluator error: {e}")
            
            return 0.5  # Default
        
        return evaluator
    
    def reason(self, problem: str, **kwargs) -> tuple[str, ThoughtTree]:
        """
        Apply ToT reasoning to a problem.
        
        Args:
            problem: Problem to solve
            **kwargs: Additional arguments for reasoner
            
        Returns:
            Tuple of (answer, thought tree)
        """
        if self._reasoner is None:
            self._reasoner = ToTReasoner(
                generator=self._create_generator(),
                evaluator=self._create_evaluator(),
                **kwargs
            )
        
        return self._reasoner.reason(problem)


# Singleton
_tot_integration: Optional[ForgeToTIntegration] = None


def get_tot_integration(engine=None) -> ForgeToTIntegration:
    """Get the ToT integration singleton."""
    global _tot_integration
    if _tot_integration is None:
        _tot_integration = ForgeToTIntegration(engine)
    elif engine:
        _tot_integration.set_engine(engine)
    return _tot_integration


__all__ = [
    'ThoughtTree',
    'ThoughtNode',
    'ToTReasoner',
    'SearchStrategy',
    'NodeState',
    'ForgeToTIntegration',
    'create_tot_reasoner',
    'get_tot_integration'
]
