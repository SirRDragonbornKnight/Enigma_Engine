"""
Multi-Agent Debate System for Enigma AI Engine

Multiple AI agents debate to reach better answers.

Features:
- Agent personas with different viewpoints
- Structured debate rounds
- Moderated debates
- Consensus building
- Fact-checking integration

Usage:
    from enigma_engine.agents.debate import DebateSystem, Agent
    
    # Quick debate
    debate = DebateSystem()
    result = debate.debate("Is AI beneficial for society?")
    
    # Custom agents
    debate.add_agent(Agent("Optimist", "Focus on positive aspects"))
    debate.add_agent(Agent("Skeptic", "Challenge assumptions"))
    result = debate.debate("Topic", rounds=3)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Standard debate roles."""
    PROPONENT = auto()    # Argues for
    OPPONENT = auto()     # Argues against
    NEUTRAL = auto()      # Balanced view
    DEVILS_ADVOCATE = auto()  # Challenges everything
    FACT_CHECKER = auto()     # Verifies claims
    MODERATOR = auto()        # Guides discussion
    SYNTHESIZER = auto()      # Combines viewpoints


@dataclass
class Agent:
    """A debating agent."""
    name: str
    persona: str = ""
    role: AgentRole = AgentRole.NEUTRAL
    
    # Agent settings
    model: Any = None
    temperature: float = 0.7
    
    # State
    arguments: List[str] = field(default_factory=list)
    score: float = 0.0
    
    def get_system_prompt(self) -> str:
        """Build system prompt for agent."""
        base = f"You are {self.name}"
        
        if self.persona:
            base += f", {self.persona}"
        
        role_prompts = {
            AgentRole.PROPONENT: "Argue in favor of the topic, highlighting benefits and positives.",
            AgentRole.OPPONENT: "Argue against the topic, highlighting risks and negatives.",
            AgentRole.NEUTRAL: "Present a balanced view, considering multiple perspectives.",
            AgentRole.DEVILS_ADVOCATE: "Challenge all assumptions and arguments from others.",
            AgentRole.FACT_CHECKER: "Verify factual claims and point out any inaccuracies.",
            AgentRole.SYNTHESIZER: "Find common ground and synthesize viewpoints."
        }
        
        if self.role in role_prompts:
            base += ". " + role_prompts[self.role]
        
        return base


@dataclass
class DebateTurn:
    """A single turn in the debate."""
    round: int
    agent: str
    argument: str
    is_rebuttal: bool = False
    references: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateResult:
    """Result of a debate."""
    topic: str
    turns: List[DebateTurn]
    consensus: Optional[str] = None
    summary: str = ""
    winner: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "turns": [
                {
                    "round": t.round,
                    "agent": t.agent,
                    "argument": t.argument,
                    "is_rebuttal": t.is_rebuttal
                }
                for t in self.turns
            ],
            "consensus": self.consensus,
            "summary": self.summary,
            "winner": self.winner,
            "scores": self.scores
        }
    
    def format_transcript(self) -> str:
        """Format as readable transcript."""
        lines = [f"=== Debate: {self.topic} ===", ""]
        
        current_round = 0
        for turn in self.turns:
            if turn.round != current_round:
                current_round = turn.round
                lines.append(f"--- Round {current_round} ---")
                lines.append("")
            
            prefix = "[REBUTTAL] " if turn.is_rebuttal else ""
            lines.append(f"{prefix}{turn.agent}:")
            lines.append(turn.argument)
            lines.append("")
        
        if self.summary:
            lines.append("=== Summary ===")
            lines.append(self.summary)
        
        if self.consensus:
            lines.append("")
            lines.append(f"Consensus: {self.consensus}")
        
        return "\n".join(lines)


class DebateSystem:
    """
    Multi-agent debate engine.
    """
    
    DEFAULT_AGENTS = [
        Agent("Alex", "a logical analyst", AgentRole.PROPONENT),
        Agent("Sam", "a critical thinker", AgentRole.OPPONENT),
        Agent("Morgan", "a balanced observer", AgentRole.NEUTRAL)
    ]
    
    OPENING_PROMPT = """Topic: {topic}

You are presenting your opening statement in a debate.
Present your main argument clearly and concisely.

Your opening statement:"""
    
    REBUTTAL_PROMPT = """Topic: {topic}

Previous arguments:
{history}

Respond to the previous arguments. Address specific points and present counter-arguments.

Your rebuttal:"""
    
    CLOSING_PROMPT = """Topic: {topic}

The debate so far:
{history}

Present your closing statement, summarizing your key points.

Your closing:"""
    
    SYNTHESIS_PROMPT = """Topic: {topic}

The full debate:
{transcript}

Synthesize the key points from all perspectives.
What are the main areas of agreement and disagreement?
Is there a possible consensus?

Synthesis:"""
    
    def __init__(
        self,
        default_model: Any = None,
        moderated: bool = True
    ):
        """
        Initialize debate system.
        
        Args:
            default_model: Default model for agents
            moderated: Whether to include a moderator
        """
        self._default_model = default_model
        self._moderated = moderated
        self._agents: List[Agent] = []
    
    def add_agent(self, agent: Agent) -> "DebateSystem":
        """Add an agent to the debate."""
        if agent.model is None:
            agent.model = self._default_model
        self._agents.append(agent)
        return self
    
    def clear_agents(self):
        """Clear all agents."""
        self._agents = []
    
    def use_default_agents(self) -> "DebateSystem":
        """Use default agent configuration."""
        self._agents = []
        for agent in self.DEFAULT_AGENTS:
            self.add_agent(Agent(
                name=agent.name,
                persona=agent.persona,
                role=agent.role,
                model=self._default_model
            ))
        return self
    
    def debate(
        self,
        topic: str,
        rounds: int = 2,
        with_closing: bool = True
    ) -> DebateResult:
        """
        Run a debate.
        
        Args:
            topic: Debate topic
            rounds: Number of rounds
            with_closing: Include closing statements
            
        Returns:
            Debate result
        """
        start_time = time.time()
        
        # Use default agents if none set
        if not self._agents:
            self.use_default_agents()
        
        turns: List[DebateTurn] = []
        
        # Opening statements (round 1)
        logger.info(f"Starting debate: {topic}")
        for agent in self._agents:
            argument = self._generate_opening(agent, topic)
            turns.append(DebateTurn(
                round=1,
                agent=agent.name,
                argument=argument
            ))
            agent.arguments.append(argument)
        
        # Rebuttal rounds
        for round_num in range(2, rounds + 1):
            history = self._format_history(turns)
            
            for agent in self._agents:
                argument = self._generate_rebuttal(agent, topic, history)
                turns.append(DebateTurn(
                    round=round_num,
                    agent=agent.name,
                    argument=argument,
                    is_rebuttal=True
                ))
                agent.arguments.append(argument)
        
        # Closing statements
        if with_closing:
            history = self._format_history(turns)
            for agent in self._agents:
                closing = self._generate_closing(agent, topic, history)
                turns.append(DebateTurn(
                    round=rounds + 1,
                    agent=agent.name,
                    argument=closing
                ))
        
        # Generate synthesis and consensus
        transcript = self._format_history(turns)
        summary, consensus = self._synthesize(topic, transcript)
        
        # Score agents
        scores = self._score_agents(turns)
        winner = max(scores.keys(), key=lambda k: scores[k]) if scores else None
        
        result = DebateResult(
            topic=topic,
            turns=turns,
            consensus=consensus,
            summary=summary,
            winner=winner,
            scores=scores,
            duration=time.time() - start_time
        )
        
        logger.info(f"Debate complete: {len(turns)} turns, {result.duration:.1f}s")
        return result
    
    def _generate(self, agent: Agent, prompt: str) -> str:
        """Generate response for an agent."""
        model = agent.model or self._default_model
        
        if model is None:
            # Mock response for testing
            return f"[{agent.name}] Mock argument for the topic."
        
        # Build full prompt with persona
        system = agent.get_system_prompt()
        full_prompt = f"{system}\n\n{prompt}"
        
        if hasattr(model, "generate"):
            return model.generate(full_prompt, temperature=agent.temperature)
        elif hasattr(model, "chat"):
            return model.chat(full_prompt)
        else:
            return str(model(full_prompt))
    
    def _generate_opening(self, agent: Agent, topic: str) -> str:
        """Generate opening statement."""
        prompt = self.OPENING_PROMPT.format(topic=topic)
        return self._generate(agent, prompt)
    
    def _generate_rebuttal(self, agent: Agent, topic: str, history: str) -> str:
        """Generate rebuttal."""
        prompt = self.REBUTTAL_PROMPT.format(topic=topic, history=history)
        return self._generate(agent, prompt)
    
    def _generate_closing(self, agent: Agent, topic: str, history: str) -> str:
        """Generate closing statement."""
        prompt = self.CLOSING_PROMPT.format(topic=topic, history=history)
        return self._generate(agent, prompt)
    
    def _synthesize(self, topic: str, transcript: str) -> tuple[str, Optional[str]]:
        """Synthesize debate and find consensus."""
        # Use first available model or return empty
        model = self._default_model
        for agent in self._agents:
            if agent.model:
                model = agent.model
                break
        
        if model is None:
            return "No synthesis available (no model)", None
        
        prompt = self.SYNTHESIS_PROMPT.format(topic=topic, transcript=transcript)
        
        if hasattr(model, "generate"):
            synthesis = model.generate(prompt)
        else:
            synthesis = str(model(prompt))
        
        # Try to extract consensus
        consensus = None
        synthesis_lower = synthesis.lower()
        
        for marker in ["consensus:", "agreement:", "common ground:"]:
            idx = synthesis_lower.find(marker)
            if idx >= 0:
                rest = synthesis[idx + len(marker):].strip()
                consensus = rest.split("\n")[0].strip()
                break
        
        return synthesis, consensus
    
    def _format_history(self, turns: List[DebateTurn]) -> str:
        """Format turns as history text."""
        lines = []
        for turn in turns:
            lines.append(f"{turn.agent}: {turn.argument}")
            lines.append("")
        return "\n".join(lines)
    
    def _score_agents(self, turns: List[DebateTurn]) -> Dict[str, float]:
        """Simple scoring based on argument length and rebuttals."""
        scores = {}
        
        for agent in self._agents:
            agent_turns = [t for t in turns if t.agent == agent.name]
            
            # Base score from number of turns
            score = len(agent_turns) * 10
            
            # Bonus for longer arguments (up to 500 chars)
            for turn in agent_turns:
                score += min(len(turn.argument), 500) / 100
            
            # Bonus for rebuttals
            rebuttals = sum(1 for t in agent_turns if t.is_rebuttal)
            score += rebuttals * 5
            
            scores[agent.name] = round(score, 1)
        
        return scores
    
    def get_agents(self) -> List[Agent]:
        """Get current agents."""
        return self._agents.copy()


class SocraticDialogue:
    """
    Socratic method dialogue for deeper understanding.
    """
    
    QUESTION_PROMPT = """Topic: {topic}
    
Current understanding: {context}

Ask a probing Socratic question that challenges assumptions or deepens understanding.

Question:"""
    
    ANSWER_PROMPT = """Topic: {topic}

Question: {question}

Previous context: {context}

Provide a thoughtful answer that advances the understanding.

Answer:"""
    
    def __init__(self, model: Any = None):
        """Initialize Socratic dialogue."""
        self._model = model
    
    def explore(
        self,
        topic: str,
        turns: int = 5,
        initial_statement: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Explore a topic through Socratic questioning.
        
        Args:
            topic: Topic to explore
            turns: Number of question-answer pairs
            initial_statement: Starting statement
            
        Returns:
            List of question-answer pairs
        """
        dialogue = []
        context = initial_statement or f"We are exploring: {topic}"
        
        for _ in range(turns):
            # Generate question
            q_prompt = self.QUESTION_PROMPT.format(topic=topic, context=context)
            question = self._generate(q_prompt)
            
            # Generate answer
            a_prompt = self.ANSWER_PROMPT.format(
                topic=topic, 
                question=question,
                context=context
            )
            answer = self._generate(a_prompt)
            
            dialogue.append({
                "question": question,
                "answer": answer
            })
            
            # Update context
            context = f"{context}\n\nQ: {question}\nA: {answer}"
        
        return dialogue
    
    def _generate(self, prompt: str) -> str:
        """Generate text."""
        if self._model is None:
            return "No model available."
        
        if hasattr(self._model, "generate"):
            return self._model.generate(prompt)
        return str(self._model(prompt))


def quick_debate(topic: str, model: Any = None, rounds: int = 2) -> DebateResult:
    """
    Quick function to run a debate.
    
    Args:
        topic: Debate topic
        model: Language model
        rounds: Number of rounds
        
    Returns:
        DebateResult
    """
    system = DebateSystem(default_model=model)
    return system.debate(topic, rounds=rounds)
