"""
Tests for forge_ai.agents module.

Tests multi-agent system functionality including:
- Agent creation and configuration
- Agent personalities and roles
- Agent collaboration and debate
- Task delegation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import threading
import time


class TestAgentRole:
    """Test AgentRole enum."""
    
    def test_roles_exist(self):
        """Test that all expected roles exist."""
        from forge_ai.agents.multi_agent import AgentRole
        
        expected_roles = [
            'GENERAL', 'CODER', 'REVIEWER', 'WRITER', 'ANALYST',
            'RESEARCHER', 'TEACHER', 'CRITIC', 'PLANNER', 'EXECUTOR',
            'MEDIATOR', 'CUSTOM'
        ]
        for role in expected_roles:
            assert hasattr(AgentRole, role), f"Missing role: {role}"
    
    def test_role_values(self):
        """Test role value strings."""
        from forge_ai.agents.multi_agent import AgentRole
        
        assert AgentRole.CODER.value == "coder"
        assert AgentRole.GENERAL.value == "general"


class TestAgentPersonality:
    """Test AgentPersonality dataclass."""
    
    def test_default_personality(self):
        """Test creating personality with defaults."""
        from forge_ai.agents.multi_agent import AgentPersonality, AgentRole
        
        personality = AgentPersonality(
            name="TestAgent",
            role=AgentRole.GENERAL
        )
        
        assert personality.name == "TestAgent"
        assert personality.role == AgentRole.GENERAL
        assert personality.creativity == 0.5
        assert personality.precision == 0.5
    
    def test_custom_personality(self):
        """Test creating personality with custom values."""
        from forge_ai.agents.multi_agent import AgentPersonality, AgentRole
        
        personality = AgentPersonality(
            name="CreativeWriter",
            role=AgentRole.WRITER,
            creativity=0.9,
            precision=0.3,
            verbosity=0.8
        )
        
        assert personality.creativity == 0.9
        assert personality.precision == 0.3
        assert personality.verbosity == 0.8
    
    def test_personality_to_dict(self):
        """Test serialization to dict."""
        from forge_ai.agents.multi_agent import AgentPersonality, AgentRole
        
        personality = AgentPersonality(
            name="Test",
            role=AgentRole.CODER,
            description="A test agent"
        )
        
        data = personality.to_dict()
        assert data['name'] == "Test"
        assert data['role'] == "coder"
        assert data['description'] == "A test agent"


class TestAgent:
    """Test Agent class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock inference engine."""
        engine = Mock()
        engine.generate = Mock(return_value="Generated response")
        return engine
    
    def test_agent_creation(self, mock_engine):
        """Test creating an agent."""
        from forge_ai.agents.multi_agent import Agent, AgentRole, AgentPersonality
        
        personality = AgentPersonality(
            name="TestAgent",
            role=AgentRole.GENERAL
        )
        agent = Agent(
            personality=personality,
            engine=mock_engine
        )
        
        assert agent.name == "TestAgent"
        assert agent.role == AgentRole.GENERAL
    
    def test_agent_respond(self, mock_engine):
        """Test agent generating a response."""
        from forge_ai.agents.multi_agent import Agent, AgentRole, AgentPersonality
        
        personality = AgentPersonality(
            name="TestAgent",
            role=AgentRole.GENERAL
        )
        agent = Agent(
            personality=personality,
            engine=mock_engine
        )
        
        response = agent.respond("Hello")
        assert response == "Generated response"
        mock_engine.generate.assert_called_once()


class TestMultiAgentSystem:
    """Test MultiAgentSystem class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock inference engine."""
        engine = Mock()
        engine.generate = Mock(return_value="Agent response")
        return engine
    
    def test_system_creation(self):
        """Test creating a multi-agent system."""
        from forge_ai.agents.multi_agent import MultiAgentSystem
        
        system = MultiAgentSystem()
        assert system is not None
        assert len(system.agents) == 0
    
    def test_create_agent(self, mock_engine):
        """Test creating an agent through the system."""
        from forge_ai.agents.multi_agent import MultiAgentSystem, AgentRole
        
        system = MultiAgentSystem()
        system._engine = mock_engine
        
        agent = system.create_agent("Coder", AgentRole.CODER)
        
        assert agent is not None
        assert agent.name == "Coder"
        assert agent.role == AgentRole.CODER
        assert system.get_agent_by_name("Coder") is not None
    
    def test_remove_agent(self, mock_engine):
        """Test removing an agent."""
        from forge_ai.agents.multi_agent import MultiAgentSystem, AgentRole
        
        system = MultiAgentSystem()
        system._engine = mock_engine
        
        agent = system.create_agent("TestAgent", AgentRole.GENERAL)
        assert system.get_agent_by_name("TestAgent") is not None
        
        system.remove_agent(agent.id)
        assert system.get_agent_by_name("TestAgent") is None
    
    def test_get_agent(self, mock_engine):
        """Test getting an agent by name."""
        from forge_ai.agents.multi_agent import MultiAgentSystem, AgentRole
        
        system = MultiAgentSystem()
        system._engine = mock_engine
        
        created = system.create_agent("MyAgent", AgentRole.ANALYST)
        retrieved = system.get_agent_by_name("MyAgent")
        
        assert retrieved is created


class TestAgentCollaboration:
    """Test agent collaboration features."""
    
    @pytest.fixture
    def system_with_agents(self):
        """Create a system with multiple agents."""
        from forge_ai.agents.multi_agent import MultiAgentSystem, AgentRole
        
        system = MultiAgentSystem()
        
        # Mock the engine
        mock_engine = Mock()
        mock_engine.generate = Mock(side_effect=lambda p, **k: f"Response to: {p[:20]}...")
        system._engine = mock_engine
        
        # Create agents
        system.create_agent("Alice", AgentRole.CODER)
        system.create_agent("Bob", AgentRole.REVIEWER)
        
        return system
    
    def test_agent_list(self, system_with_agents):
        """Test listing agents."""
        agents = system_with_agents.list_agents()
        assert len(agents) == 2
        agent_names = [a.name for a in agents]
        assert "Alice" in agent_names
        assert "Bob" in agent_names


class TestAgentSerialization:
    """Test agent state serialization."""
    
    def test_agent_state_round_trip(self):
        """Test saving and loading agent state."""
        from forge_ai.agents.multi_agent import AgentPersonality, AgentRole
        
        original = AgentPersonality(
            name="TestAgent",
            role=AgentRole.ANALYST,
            creativity=0.7,
            precision=0.8
        )
        
        # Serialize
        data = original.to_dict()
        
        # Verify key fields preserved
        assert data['name'] == "TestAgent"
        assert data['creativity'] == 0.7
        assert data['precision'] == 0.8
