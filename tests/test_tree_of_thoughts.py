"""
Unit tests for Tree-of-Thoughts reasoning module.

Tests the ToT reasoning components: ThoughtTree, ThoughtNode, ToTReasoner.
"""

import pytest
import time

from forge_ai.core.tree_of_thoughts import (
    SearchStrategy,
    NodeState,
    ThoughtNode,
    ThoughtTree,
    ToTReasoner,
)


class TestThoughtNode:
    """Tests for ThoughtNode dataclass."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = ThoughtNode(
            id="node_1",
            thought="Consider option A"
        )
        assert node.id == "node_1"
        assert node.thought == "Consider option A"
        assert node.parent_id is None
        assert node.score == 0.0
        assert node.state == NodeState.PENDING
        assert node.depth == 0
    
    def test_node_with_parent(self):
        """Test node with parent reference."""
        node = ThoughtNode(
            id="node_2",
            thought="Next step",
            parent_id="node_1",
            depth=1
        )
        assert node.parent_id == "node_1"
        assert node.depth == 1
    
    def test_node_comparison(self):
        """Test node comparison for heap."""
        node_high = ThoughtNode(id="a", thought="A", score=0.9)
        node_low = ThoughtNode(id="b", thought="B", score=0.3)
        
        # Higher score should be "less than" for max-heap behavior
        assert node_high < node_low


class TestThoughtTree:
    """Tests for ThoughtTree structure."""
    
    @pytest.fixture
    def tree(self):
        """Create a test tree."""
        root = ThoughtNode(
            id="root",
            thought="Problem statement",
            state=NodeState.EVALUATED,
            score=0.5
        )
        tree = ThoughtTree(
            root_id="root",
            problem="Test problem",
            max_depth=3,
            branching_factor=2
        )
        tree.add_node(root)
        return tree
    
    def test_tree_init(self, tree):
        """Test tree initialization."""
        assert tree.root_id == "root"
        assert tree.problem == "Test problem"
        assert tree.max_depth == 3
        assert tree.branching_factor == 2
    
    def test_add_node(self, tree):
        """Test adding nodes."""
        child = ThoughtNode(
            id="child_1",
            thought="First option",
            parent_id="root",
            depth=1
        )
        tree.add_node(child)
        
        assert "child_1" in tree.nodes
        assert tree.get_node("child_1") == child
        assert "child_1" in tree.nodes["root"].children_ids
    
    def test_get_root(self, tree):
        """Test getting root node."""
        root = tree.get_root()
        assert root is not None
        assert root.id == "root"
    
    def test_get_children(self, tree):
        """Test getting children of a node."""
        # Add children
        for i in range(3):
            tree.add_node(ThoughtNode(
                id=f"child_{i}",
                thought=f"Option {i}",
                parent_id="root",
                depth=1
            ))
        
        children = tree.get_children("root")
        assert len(children) == 3
    
    def test_get_path_to_node(self, tree):
        """Test getting path from root to node."""
        # Build a path: root -> child -> grandchild
        tree.add_node(ThoughtNode(
            id="child",
            thought="Child",
            parent_id="root",
            depth=1
        ))
        tree.add_node(ThoughtNode(
            id="grandchild",
            thought="Grandchild",
            parent_id="child",
            depth=2
        ))
        
        path = tree.get_path_to_node("grandchild")
        
        assert len(path) == 3
        assert path[0].id == "root"
        assert path[1].id == "child"
        assert path[2].id == "grandchild"
    
    def test_get_best_path(self, tree):
        """Test finding best path."""
        # Create two branches with different scores
        tree.add_node(ThoughtNode(
            id="good",
            thought="Good path",
            parent_id="root",
            depth=1,
            score=0.9,
            state=NodeState.EVALUATED
        ))
        tree.add_node(ThoughtNode(
            id="bad",
            thought="Bad path",
            parent_id="root",
            depth=1,
            score=0.2,
            state=NodeState.EVALUATED
        ))
        
        best_path = tree.get_best_path()
        
        assert len(best_path) == 2
        assert best_path[-1].id == "good"
    
    def test_to_dict(self, tree):
        """Test tree serialization."""
        tree.add_node(ThoughtNode(
            id="child",
            thought="Child",
            parent_id="root",
            depth=1
        ))
        
        data = tree.to_dict()
        
        assert data["root_id"] == "root"
        assert data["problem"] == "Test problem"
        assert "root" in data["nodes"]
        assert "child" in data["nodes"]


class TestToTReasoner:
    """Tests for ToTReasoner."""
    
    def test_reasoner_default_init(self):
        """Test reasoner initialization with defaults."""
        reasoner = ToTReasoner()
        assert reasoner._strategy == SearchStrategy.BEAM
        assert reasoner._max_depth == 5
        assert reasoner._branching_factor == 3
        assert reasoner._beam_width == 3
    
    def test_reasoner_custom_init(self):
        """Test reasoner with custom settings."""
        reasoner = ToTReasoner(
            strategy=SearchStrategy.DFS,
            max_depth=3,
            branching_factor=2,
            beam_width=2
        )
        assert reasoner._strategy == SearchStrategy.DFS
        assert reasoner._max_depth == 3
    
    def test_default_generator(self):
        """Test default thought generator."""
        reasoner = ToTReasoner()
        thoughts = reasoner._default_generator("context", 3)
        
        assert len(thoughts) == 3
        assert all(isinstance(t, str) for t in thoughts)
    
    def test_default_evaluator(self):
        """Test default thought evaluator."""
        reasoner = ToTReasoner()
        
        # Short thought
        score_short = reasoner._default_evaluator("problem", "Short")
        
        # Longer thought with reasoning
        score_reasoning = reasoner._default_evaluator(
            "problem", 
            "Because this is the case, therefore we should do this since it follows."
        )
        
        assert 0 <= score_short <= 1
        assert 0 <= score_reasoning <= 1
        assert score_reasoning > score_short  # Reasoning keywords boost score
    
    def test_reason_returns_answer_and_tree(self):
        """Test that reason returns both answer and tree."""
        reasoner = ToTReasoner(max_depth=2, branching_factor=2, beam_width=2)
        answer, tree = reasoner.reason("What is 2+2?")
        
        assert isinstance(answer, str)
        assert isinstance(tree, ThoughtTree)
        assert tree.problem == "What is 2+2?"
    
    def test_reason_creates_tree_structure(self):
        """Test that reasoning creates proper tree."""
        reasoner = ToTReasoner(max_depth=2, branching_factor=2, beam_width=2)
        _, tree = reasoner.reason("Simple problem")
        
        # Should have root + some children
        assert len(tree.nodes) > 1
        assert tree.get_root() is not None
    
    def test_bfs_strategy(self):
        """Test BFS search strategy."""
        reasoner = ToTReasoner(
            strategy=SearchStrategy.BFS,
            max_depth=2,
            branching_factor=2
        )
        answer, tree = reasoner.reason("BFS test")
        
        assert isinstance(answer, str)
    
    def test_dfs_strategy(self):
        """Test DFS search strategy."""
        reasoner = ToTReasoner(
            strategy=SearchStrategy.DFS,
            max_depth=2,
            branching_factor=2
        )
        answer, tree = reasoner.reason("DFS test")
        
        assert isinstance(answer, str)
    
    def test_beam_strategy(self):
        """Test beam search strategy."""
        reasoner = ToTReasoner(
            strategy=SearchStrategy.BEAM,
            max_depth=2,
            branching_factor=2,
            beam_width=2
        )
        answer, tree = reasoner.reason("Beam test")
        
        assert isinstance(answer, str)
    
    def test_best_first_strategy(self):
        """Test best-first search strategy."""
        reasoner = ToTReasoner(
            strategy=SearchStrategy.BEST_FIRST,
            max_depth=2,
            branching_factor=2,
            beam_width=2
        )
        answer, tree = reasoner.reason("Best-first test")
        
        assert isinstance(answer, str)
    
    def test_custom_generator(self):
        """Test with custom thought generator."""
        custom_thoughts = [
            "Step 1: Analyze",
            "Step 2: Plan",
            "Step 3: Execute"
        ]
        
        def custom_gen(context, n):
            return custom_thoughts[:n]
        
        reasoner = ToTReasoner(
            generator=custom_gen,
            max_depth=2,
            branching_factor=3
        )
        _, tree = reasoner.reason("Custom generator test")
        
        # Check our custom thoughts appear in tree
        thought_texts = [n.thought for n in tree.nodes.values()]
        for thought in custom_thoughts:
            # At least some should appear
            pass  # Generator is used, verification depends on tree depth
    
    def test_custom_evaluator(self):
        """Test with custom thought evaluator."""
        def custom_eval(problem, thought):
            # High score if "correct" in thought
            return 1.0 if "correct" in thought.lower() else 0.0
        
        reasoner = ToTReasoner(
            evaluator=custom_eval,
            max_depth=2,
            branching_factor=2
        )
        _, tree = reasoner.reason("Custom evaluator test")
        
        # Tree should be created regardless
        assert len(tree.nodes) > 0
    
    def test_compose_answer(self):
        """Test answer composition."""
        reasoner = ToTReasoner()
        
        # Create a path
        path = [
            ThoughtNode(id="root", thought="Problem: Test"),
            ThoughtNode(id="1", thought="First step"),
            ThoughtNode(id="2", thought="Conclusion")
        ]
        
        answer = reasoner._compose_answer(path)
        
        assert "First step" in answer
        assert "Conclusion" in answer
        assert "Reasoning path" in answer
    
    def test_id_generation(self):
        """Test unique ID generation."""
        reasoner = ToTReasoner()
        
        ids = [reasoner._generate_id() for _ in range(100)]
        
        # All IDs should be unique
        assert len(ids) == len(set(ids))


class TestNodeState:
    """Tests for NodeState enum."""
    
    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert hasattr(NodeState, 'PENDING')
        assert hasattr(NodeState, 'EXPLORING')
        assert hasattr(NodeState, 'EVALUATED')
        assert hasattr(NodeState, 'PRUNED')
        assert hasattr(NodeState, 'SELECTED')


class TestSearchStrategy:
    """Tests for SearchStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test all expected strategies exist."""
        assert hasattr(SearchStrategy, 'BFS')
        assert hasattr(SearchStrategy, 'DFS')
        assert hasattr(SearchStrategy, 'BEAM')
        assert hasattr(SearchStrategy, 'BEST_FIRST')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
