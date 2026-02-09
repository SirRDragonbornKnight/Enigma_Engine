"""
Agent Voting System for Enigma AI Engine

Multi-agent collaboration with voting mechanisms.

Features:
- Multiple agent personas
- Voting protocols
- Consensus building
- Debate mechanism
- Delegation

Usage:
    from enigma_engine.agents.voting import AgentVoting, get_voting_system
    
    voting = get_voting_system()
    
    # Add agents
    voting.add_agent("analyst", model1, role="analyzer")
    voting.add_agent("critic", model2, role="critic")
    
    # Get consensus
    result = voting.vote("Should we proceed with plan A?")
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class VoteType(Enum):
    """Types of votes."""
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"
    OTHER = "other"


class VotingProtocol(Enum):
    """Voting protocols."""
    MAJORITY = "majority"  # Simple majority wins
    SUPERMAJORITY = "supermajority"  # 2/3 required
    UNANIMOUS = "unanimous"  # All must agree
    RANKED_CHOICE = "ranked_choice"  # Ranked preferences
    BORDA_COUNT = "borda_count"  # Points for ranking
    APPROVAL = "approval"  # Approve multiple options


class DebatePhase(Enum):
    """Phases of debate."""
    OPENING = "opening"  # Initial statements
    ARGUMENT = "argument"  # Present arguments
    REBUTTAL = "rebuttal"  # Counter arguments
    CLOSING = "closing"  # Final statements
    VOTING = "voting"  # Final vote


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: str  # "leader", "analyst", "critic", "executor", "voter"
    
    # Voting behavior
    vote_weight: float = 1.0
    can_veto: bool = False
    required_for_quorum: bool = True
    
    # Capabilities
    can_propose: bool = True
    can_amend: bool = True
    can_delegate: bool = True


@dataclass
class Agent:
    """An agent in the voting system."""
    config: AgentConfig
    model: Any
    
    # State
    current_vote: Optional[VoteType] = None
    vote_reasoning: Optional[str] = None
    delegate_to: Optional[str] = None  # Agent name to delegate to


@dataclass
class Proposal:
    """A proposal for voting."""
    id: str
    text: str
    proposer: str
    
    # Amendments
    amendments: List[str] = field(default_factory=list)
    
    # Options (for multiple choice)
    options: List[str] = field(default_factory=list)
    
    # Metadata
    discussion: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class VoteResult:
    """Result of a vote."""
    proposal_id: str
    passed: bool
    
    # Counts
    yes_votes: int = 0
    no_votes: int = 0
    abstain_votes: int = 0
    
    # Details
    vote_breakdown: Dict[str, VoteType] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    
    # For ranked choice
    winner: Optional[str] = None
    rounds: List[Dict[str, int]] = field(default_factory=list)
    
    # Metadata
    protocol_used: str = ""
    quorum_met: bool = True


class ConsensusBuilder:
    """Build consensus among agents."""
    
    def __init__(self, agents: Dict[str, Agent]):
        self._agents = agents
    
    def find_common_ground(
        self,
        statements: Dict[str, str]
    ) -> List[str]:
        """
        Find common themes in agent statements.
        
        Returns:
            List of common points
        """
        # Simple word overlap approach
        all_words: Dict[str, int] = {}
        
        for text in statements.values():
            words = set(text.lower().split())
            for word in words:
                all_words[word] = all_words.get(word, 0) + 1
        
        # Find words mentioned by majority
        threshold = len(statements) / 2
        common_words = [w for w, c in all_words.items() if c >= threshold]
        
        # Group into themes
        themes = []
        if common_words:
            themes.append(f"Common points: {', '.join(common_words[:10])}")
        
        return themes
    
    def identify_disagreements(
        self,
        statements: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """
        Identify disagreements between agents.
        
        Returns:
            List of (agent1, agent2, topic) tuples
        """
        disagreements = []
        
        agent_names = list(statements.keys())
        
        for i, name1 in enumerate(agent_names):
            for name2 in agent_names[i+1:]:
                text1 = statements[name1].lower()
                text2 = statements[name2].lower()
                
                # Check for opposing indicators
                opposites = [
                    ("yes", "no"),
                    ("agree", "disagree"),
                    ("support", "oppose"),
                    ("correct", "incorrect"),
                    ("true", "false")
                ]
                
                for pos, neg in opposites:
                    if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                        disagreements.append((name1, name2, f"{pos}/{neg}"))
        
        return disagreements


class DebateEngine:
    """Facilitate debates between agents."""
    
    def __init__(
        self,
        agents: Dict[str, Agent],
        max_rounds: int = 3
    ):
        self._agents = agents
        self._max_rounds = max_rounds
    
    def run_debate(
        self,
        topic: str,
        positions: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[str]]:
        """
        Run a structured debate.
        
        Args:
            topic: Debate topic
            positions: Initial positions for each agent
            
        Returns:
            Dict of agent_name -> list of statements
        """
        debate_log: Dict[str, List[str]] = {
            name: [] for name in self._agents
        }
        
        # Opening statements
        for name, agent in self._agents.items():
            initial_pos = positions.get(name, "") if positions else ""
            statement = self._get_agent_response(
                agent,
                f"Opening statement on: {topic}",
                context=initial_pos
            )
            debate_log[name].append(f"[OPENING] {statement}")
        
        # Debate rounds
        for round_num in range(self._max_rounds):
            # Each agent can respond to others
            for name, agent in self._agents.items():
                # Get other statements from this round
                others = [
                    f"{other_name}: {debate_log[other_name][-1]}"
                    for other_name in self._agents
                    if other_name != name and debate_log[other_name]
                ]
                
                if others:
                    context = "\n".join(others)
                    response = self._get_agent_response(
                        agent,
                        f"Round {round_num + 1} response on: {topic}",
                        context=context
                    )
                    debate_log[name].append(f"[ROUND {round_num + 1}] {response}")
        
        # Closing statements
        for name, agent in self._agents.items():
            statement = self._get_agent_response(
                agent,
                f"Closing statement on: {topic}",
                context="\n".join(debate_log[name])
            )
            debate_log[name].append(f"[CLOSING] {statement}")
        
        return debate_log
    
    def _get_agent_response(
        self,
        agent: Agent,
        prompt: str,
        context: str = ""
    ) -> str:
        """Get response from agent."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        try:
            if hasattr(agent.model, 'generate'):
                response = agent.model.generate(full_prompt)
                return str(response)[:500]  # Limit length
            elif hasattr(agent.model, 'chat'):
                response = agent.model.chat(full_prompt)
                return str(response)[:500]
            elif callable(agent.model):
                response = agent.model(full_prompt)
                return str(response)[:500]
            else:
                return f"[{agent.config.role}] Response to: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"Agent response failed: {e}")
            return f"[{agent.config.role}] (Error generating response)"


class AgentVoting:
    """Multi-agent voting system."""
    
    def __init__(
        self,
        protocol: VotingProtocol = VotingProtocol.MAJORITY,
        quorum_threshold: float = 0.5
    ):
        """
        Initialize voting system.
        
        Args:
            protocol: Default voting protocol
            quorum_threshold: Minimum participation
        """
        self._agents: Dict[str, Agent] = {}
        self._protocol = protocol
        self._quorum_threshold = quorum_threshold
        
        self._proposals: Dict[str, Proposal] = {}
        self._results: Dict[str, VoteResult] = {}
        
        self._consensus = ConsensusBuilder(self._agents)
        self._debate = DebateEngine(self._agents)
    
    def add_agent(
        self,
        name: str,
        model: Any,
        role: str = "voter",
        vote_weight: float = 1.0,
        can_veto: bool = False
    ):
        """Add an agent to the voting system."""
        config = AgentConfig(
            name=name,
            role=role,
            vote_weight=vote_weight,
            can_veto=can_veto
        )
        
        self._agents[name] = Agent(config=config, model=model)
        
        # Rebuild helpers
        self._consensus = ConsensusBuilder(self._agents)
        self._debate = DebateEngine(self._agents)
        
        logger.info(f"Added agent '{name}' with role '{role}'")
    
    def remove_agent(self, name: str):
        """Remove an agent."""
        if name in self._agents:
            del self._agents[name]
            self._consensus = ConsensusBuilder(self._agents)
            self._debate = DebateEngine(self._agents)
    
    def create_proposal(
        self,
        proposal_id: str,
        text: str,
        proposer: str,
        options: Optional[List[str]] = None
    ) -> Proposal:
        """
        Create a proposal for voting.
        
        Returns:
            The proposal
        """
        proposal = Proposal(
            id=proposal_id,
            text=text,
            proposer=proposer,
            options=options or []
        )
        
        self._proposals[proposal_id] = proposal
        return proposal
    
    def vote(
        self,
        question: str,
        options: Optional[List[str]] = None,
        protocol: Optional[VotingProtocol] = None,
        with_debate: bool = False
    ) -> VoteResult:
        """
        Conduct a vote.
        
        Args:
            question: Question to vote on
            options: Options (if not yes/no)
            protocol: Override voting protocol
            with_debate: Run debate first
            
        Returns:
            VoteResult with outcome
        """
        protocol = protocol or self._protocol
        
        # Create proposal
        proposal_id = f"proposal_{len(self._proposals)}"
        proposal = self.create_proposal(proposal_id, question, "system", options)
        
        # Run debate if requested
        if with_debate:
            debate_log = self._debate.run_debate(question)
            for name, statements in debate_log.items():
                for statement in statements:
                    proposal.discussion.append({
                        "agent": name,
                        "statement": statement
                    })
        
        # Collect votes
        votes: Dict[str, VoteType] = {}
        reasoning: Dict[str, str] = {}
        
        for name, agent in self._agents.items():
            # Handle delegation
            if agent.delegate_to and agent.delegate_to in self._agents:
                delegated = self._agents[agent.delegate_to]
                vote, reason = self._get_agent_vote(delegated, question, options)
            else:
                vote, reason = self._get_agent_vote(agent, question, options)
            
            votes[name] = vote
            reasoning[name] = reason
            agent.current_vote = vote
            agent.vote_reasoning = reason
        
        # Calculate result based on protocol
        if protocol == VotingProtocol.MAJORITY:
            result = self._majority_result(proposal, votes, reasoning)
        elif protocol == VotingProtocol.SUPERMAJORITY:
            result = self._supermajority_result(proposal, votes, reasoning)
        elif protocol == VotingProtocol.UNANIMOUS:
            result = self._unanimous_result(proposal, votes, reasoning)
        elif protocol == VotingProtocol.RANKED_CHOICE:
            # Requires options
            result = self._ranked_choice_result(proposal, votes, reasoning, options or [])
        else:
            result = self._majority_result(proposal, votes, reasoning)
        
        result.protocol_used = protocol.value
        
        # Check veto
        for name, agent in self._agents.items():
            if agent.config.can_veto and votes.get(name) == VoteType.NO:
                result.passed = False
                result.reasoning[name] += " [VETO]"
        
        # Store result
        self._results[proposal_id] = result
        
        return result
    
    def _get_agent_vote(
        self,
        agent: Agent,
        question: str,
        options: Optional[List[str]]
    ) -> Tuple[VoteType, str]:
        """Get vote from an agent."""
        prompt = f"Vote on: {question}\n"
        
        if options:
            prompt += f"Options: {', '.join(options)}\n"
        else:
            prompt += "Options: yes, no, abstain\n"
        
        prompt += "Respond with your vote and brief reasoning."
        
        try:
            if hasattr(agent.model, 'generate'):
                response = str(agent.model.generate(prompt))
            elif hasattr(agent.model, 'chat'):
                response = str(agent.model.chat(prompt))
            elif callable(agent.model):
                response = str(agent.model(prompt))
            else:
                # Random vote for dummy models
                response = random.choice(["yes", "no", "abstain"])
                return self._parse_vote(response), f"Random: {response}"
            
            # Parse vote from response
            vote = self._parse_vote(response)
            return vote, response[:200]
            
        except Exception as e:
            logger.error(f"Agent vote failed: {e}")
            return VoteType.ABSTAIN, f"Error: {e}"
    
    def _parse_vote(self, response: str) -> VoteType:
        """Parse vote from response text."""
        response_lower = response.lower()
        
        if any(w in response_lower for w in ["yes", "approve", "agree", "support", "aye"]):
            return VoteType.YES
        elif any(w in response_lower for w in ["no", "reject", "disagree", "oppose", "nay"]):
            return VoteType.NO
        elif any(w in response_lower for w in ["abstain", "skip", "neutral"]):
            return VoteType.ABSTAIN
        else:
            return VoteType.OTHER
    
    def _majority_result(
        self,
        proposal: Proposal,
        votes: Dict[str, VoteType],
        reasoning: Dict[str, str]
    ) -> VoteResult:
        """Calculate majority vote result."""
        yes_count = sum(1 for v in votes.values() if v == VoteType.YES)
        no_count = sum(1 for v in votes.values() if v == VoteType.NO)
        abstain_count = sum(1 for v in votes.values() if v == VoteType.ABSTAIN)
        
        # Weight votes
        yes_weight = sum(
            self._agents[n].config.vote_weight
            for n, v in votes.items()
            if v == VoteType.YES
        )
        no_weight = sum(
            self._agents[n].config.vote_weight
            for n, v in votes.items()
            if v == VoteType.NO
        )
        
        passed = yes_weight > no_weight
        
        return VoteResult(
            proposal_id=proposal.id,
            passed=passed,
            yes_votes=yes_count,
            no_votes=no_count,
            abstain_votes=abstain_count,
            vote_breakdown=votes,
            reasoning=reasoning
        )
    
    def _supermajority_result(
        self,
        proposal: Proposal,
        votes: Dict[str, VoteType],
        reasoning: Dict[str, str]
    ) -> VoteResult:
        """Calculate supermajority (2/3) result."""
        result = self._majority_result(proposal, votes, reasoning)
        
        # Require 2/3 yes
        total_voting = result.yes_votes + result.no_votes
        if total_voting > 0:
            yes_ratio = result.yes_votes / total_voting
            result.passed = yes_ratio >= 0.667
        else:
            result.passed = False
        
        return result
    
    def _unanimous_result(
        self,
        proposal: Proposal,
        votes: Dict[str, VoteType],
        reasoning: Dict[str, str]
    ) -> VoteResult:
        """Calculate unanimous result."""
        result = self._majority_result(proposal, votes, reasoning)
        
        # All must be yes (abstains OK)
        non_abstain = [v for v in votes.values() if v != VoteType.ABSTAIN]
        result.passed = all(v == VoteType.YES for v in non_abstain) and len(non_abstain) > 0
        
        return result
    
    def _ranked_choice_result(
        self,
        proposal: Proposal,
        votes: Dict[str, VoteType],
        reasoning: Dict[str, str],
        options: List[str]
    ) -> VoteResult:
        """Calculate ranked choice result."""
        # Simplified: just count mentions
        option_counts = Counter()
        
        for name, reason in reasoning.items():
            for option in options:
                if option.lower() in reason.lower():
                    option_counts[option] += self._agents[name].config.vote_weight
        
        winner = option_counts.most_common(1)[0][0] if option_counts else None
        
        return VoteResult(
            proposal_id=proposal.id,
            passed=winner is not None,
            winner=winner,
            vote_breakdown=votes,
            reasoning=reasoning,
            rounds=[dict(option_counts)]
        )
    
    def delegate_vote(
        self,
        from_agent: str,
        to_agent: str
    ):
        """Delegate voting from one agent to another."""
        if from_agent in self._agents and to_agent in self._agents:
            self._agents[from_agent].delegate_to = to_agent
    
    def run_debate(
        self,
        topic: str
    ) -> Dict[str, List[str]]:
        """Run a debate on a topic."""
        return self._debate.run_debate(topic)
    
    def find_consensus(
        self,
        topic: str
    ) -> Dict[str, Any]:
        """
        Find consensus on a topic.
        
        Returns:
            Consensus information
        """
        # Get initial positions
        positions = {}
        for name, agent in self._agents.items():
            prompt = f"What is your position on: {topic}"
            try:
                if hasattr(agent.model, 'generate'):
                    positions[name] = str(agent.model.generate(prompt))[:300]
                elif callable(agent.model):
                    positions[name] = str(agent.model(prompt))[:300]
                else:
                    positions[name] = f"[{agent.config.role}] Position on {topic}"
            except Exception:
                positions[name] = ""
        
        common = self._consensus.find_common_ground(positions)
        disagreements = self._consensus.identify_disagreements(positions)
        
        return {
            "positions": positions,
            "common_ground": common,
            "disagreements": disagreements
        }
    
    def get_results(self) -> Dict[str, VoteResult]:
        """Get all voting results."""
        return self._results
    
    def list_agents(self) -> List[str]:
        """List all agents."""
        return list(self._agents.keys())


# Global instance
_voting: Optional[AgentVoting] = None


def get_voting_system(
    protocol: VotingProtocol = VotingProtocol.MAJORITY
) -> AgentVoting:
    """Get or create global voting system."""
    global _voting
    if _voting is None:
        _voting = AgentVoting(protocol)
    return _voting
