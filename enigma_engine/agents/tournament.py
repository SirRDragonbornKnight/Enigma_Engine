"""
Agent Tournament System for Enigma AI Engine

Run tournaments between multiple agents.

Features:
- Round-robin tournaments
- Single/double elimination
- ELO rating system
- Bracket visualization
- Match history

Usage:
    from enigma_engine.agents.tournament import Tournament, TournamentType
    
    tournament = Tournament(TournamentType.SINGLE_ELIMINATION)
    
    # Add agents
    tournament.add_participant("agent1", AgentA())
    tournament.add_participant("agent2", AgentB())
    tournament.add_participant("agent3", AgentC())
    tournament.add_participant("agent4", AgentD())
    
    # Run tournament
    result = tournament.run("Code golf challenge")
"""

import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TournamentType(Enum):
    """Tournament formats."""
    ROUND_ROBIN = "round_robin"
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    SWISS = "swiss"


class MatchResult(Enum):
    """Match outcomes."""
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"
    FORFEIT = "forfeit"


@dataclass
class Participant:
    """Tournament participant."""
    participant_id: str
    name: str
    agent: Any
    elo_rating: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    eliminated: bool = False


@dataclass
class Match:
    """A single match between participants."""
    match_id: str
    participant_a: str
    participant_b: str
    round_num: int
    winner: Optional[str] = None
    score_a: float = 0.0
    score_b: float = 0.0
    result: Optional[MatchResult] = None
    timestamp: float = field(default_factory=time.time)
    bracket_position: Tuple[int, int] = (0, 0)


@dataclass
class TournamentResult:
    """Tournament result."""
    tournament_id: str
    tournament_type: TournamentType
    challenge: str
    champion: Optional[str]
    final_standings: List[Tuple[str, int]]
    matches: List[Match]
    elo_changes: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class EloSystem:
    """ELO rating system."""
    
    K_FACTOR = 32  # Standard K-factor
    
    @classmethod
    def expected_score(cls, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    @classmethod
    def update_ratings(
        cls,
        rating_a: float,
        rating_b: float,
        score_a: float,  # 1 for win, 0 for loss, 0.5 for draw
        score_b: float
    ) -> Tuple[float, float]:
        """
        Update ELO ratings after a match.
        
        Returns:
            New ratings for A and B
        """
        expected_a = cls.expected_score(rating_a, rating_b)
        expected_b = cls.expected_score(rating_b, rating_a)
        
        new_rating_a = rating_a + cls.K_FACTOR * (score_a - expected_a)
        new_rating_b = rating_b + cls.K_FACTOR * (score_b - expected_b)
        
        return new_rating_a, new_rating_b


class MatchJudge:
    """Judge matches between agents."""
    
    def __init__(self, scoring_fn: Optional[Callable] = None):
        """
        Initialize judge.
        
        Args:
            scoring_fn: Custom scoring function (agent, challenge) -> score
        """
        self.scoring_fn = scoring_fn
    
    def judge_match(
        self,
        participant_a: Participant,
        participant_b: Participant,
        challenge: str
    ) -> Match:
        """
        Judge a match between two participants.
        
        Args:
            participant_a: First participant
            participant_b: Second participant
            challenge: The challenge/task
            
        Returns:
            Match result
        """
        match = Match(
            match_id=str(uuid.uuid4())[:8],
            participant_a=participant_a.participant_id,
            participant_b=participant_b.participant_id,
            round_num=0
        )
        
        # Get scores
        score_a = self._get_score(participant_a.agent, challenge)
        score_b = self._get_score(participant_b.agent, challenge)
        
        match.score_a = score_a
        match.score_b = score_b
        
        # Determine winner
        if score_a > score_b:
            match.winner = participant_a.participant_id
            match.result = MatchResult.WIN
        elif score_b > score_a:
            match.winner = participant_b.participant_id
            match.result = MatchResult.WIN
        else:
            match.winner = None
            match.result = MatchResult.DRAW
        
        return match
    
    def _get_score(self, agent: Any, challenge: str) -> float:
        """Get score for an agent."""
        if self.scoring_fn:
            try:
                return float(self.scoring_fn(agent, challenge))
            except Exception as e:
                logger.error(f"Scoring error: {e}")
                return 0.0
        
        # Default scoring based on agent response
        if agent is None:
            return random.random() * 10
        
        try:
            if hasattr(agent, 'solve'):
                result = agent.solve(challenge)
            elif hasattr(agent, 'execute'):
                result = agent.execute(challenge)
            elif hasattr(agent, '__call__'):
                result = agent(challenge)
            else:
                result = str(agent)
            
            # Score based on response length (simple heuristic)
            return min(10.0, len(str(result)) / 100)
            
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return 0.0


class Tournament:
    """Tournament manager."""
    
    def __init__(
        self,
        tournament_type: TournamentType = TournamentType.SINGLE_ELIMINATION,
        judge: Optional[MatchJudge] = None
    ):
        """
        Initialize tournament.
        
        Args:
            tournament_type: Tournament format
            judge: Match judge
        """
        self.tournament_id = str(uuid.uuid4())[:8]
        self.tournament_type = tournament_type
        self.judge = judge or MatchJudge()
        
        self._participants: Dict[str, Participant] = {}
        self._matches: List[Match] = []
        self._listeners: List[Callable[[str, Any], None]] = []
        self._history: List[TournamentResult] = []
        self._max_history: int = 100
    
    def add_participant(
        self,
        participant_id: str,
        agent: Any,
        name: Optional[str] = None,
        initial_elo: float = 1500.0
    ):
        """Add a participant."""
        self._participants[participant_id] = Participant(
            participant_id=participant_id,
            name=name or participant_id,
            agent=agent,
            elo_rating=initial_elo
        )
        self._emit("participant_added", {"id": participant_id})
    
    def remove_participant(self, participant_id: str):
        """Remove a participant."""
        if participant_id in self._participants:
            del self._participants[participant_id]
    
    def run(self, challenge: str) -> TournamentResult:
        """
        Run the tournament.
        
        Args:
            challenge: The challenge/task for the tournament
            
        Returns:
            Tournament result
        """
        if len(self._participants) < 2:
            raise ValueError("Need at least 2 participants")
        
        self._emit("tournament_started", {
            "type": self.tournament_type.value,
            "participants": len(self._participants)
        })
        
        # Reset state
        self._matches = []
        for p in self._participants.values():
            p.wins = 0
            p.losses = 0
            p.draws = 0
            p.eliminated = False
        
        # Record initial ELO
        initial_elo = {p.participant_id: p.elo_rating for p in self._participants.values()}
        
        # Run tournament based on type
        if self.tournament_type == TournamentType.ROUND_ROBIN:
            champion = self._run_round_robin(challenge)
        elif self.tournament_type == TournamentType.SINGLE_ELIMINATION:
            champion = self._run_single_elimination(challenge)
        elif self.tournament_type == TournamentType.DOUBLE_ELIMINATION:
            champion = self._run_double_elimination(challenge)
        elif self.tournament_type == TournamentType.SWISS:
            champion = self._run_swiss(challenge)
        else:
            champion = self._run_round_robin(challenge)
        
        # Calculate ELO changes
        elo_changes = {}
        for p_id, participant in self._participants.items():
            elo_changes[p_id] = participant.elo_rating - initial_elo[p_id]
        
        # Final standings
        standings = sorted(
            [(p.participant_id, p.wins) for p in self._participants.values()],
            key=lambda x: -x[1]
        )
        
        result = TournamentResult(
            tournament_id=self.tournament_id,
            tournament_type=self.tournament_type,
            challenge=challenge,
            champion=champion,
            final_standings=standings,
            matches=self._matches,
            elo_changes=elo_changes
        )
        
        self._history.append(result)
        
        # Trim history if too long
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        self._emit("tournament_ended", {
            "champion": champion,
            "matches": len(self._matches)
        })
        
        return result
    
    def _run_round_robin(self, challenge: str) -> Optional[str]:
        """Run round-robin tournament."""
        participants = list(self._participants.values())
        round_num = 1
        
        # Each participant plays every other participant
        for i, p_a in enumerate(participants):
            for p_b in participants[i+1:]:
                match = self._run_match(p_a, p_b, challenge, round_num)
                round_num += 1
        
        # Winner is participant with most wins
        winner = max(participants, key=lambda p: (p.wins, -p.losses))
        return winner.participant_id
    
    def _run_single_elimination(self, challenge: str) -> Optional[str]:
        """Run single elimination tournament."""
        participants = list(self._participants.values())
        random.shuffle(participants)
        
        # Pad to power of 2
        n = len(participants)
        bracket_size = 2 ** math.ceil(math.log2(n))
        
        # Add byes
        byes_needed = bracket_size - n
        
        current_round = participants.copy()
        round_num = 1
        
        while len(current_round) > 1:
            next_round = []
            
            for i in range(0, len(current_round), 2):
                if i + 1 >= len(current_round):
                    # Bye
                    next_round.append(current_round[i])
                    continue
                
                p_a = current_round[i]
                p_b = current_round[i + 1]
                
                match = self._run_match(p_a, p_b, challenge, round_num)
                
                if match.winner == p_a.participant_id:
                    next_round.append(p_a)
                    p_b.eliminated = True
                else:
                    next_round.append(p_b)
                    p_a.eliminated = True
            
            current_round = next_round
            round_num += 1
        
        return current_round[0].participant_id if current_round else None
    
    def _run_double_elimination(self, challenge: str) -> Optional[str]:
        """Run double elimination tournament."""
        participants = list(self._participants.values())
        random.shuffle(participants)
        
        winners_bracket = participants.copy()
        losers_bracket = []
        round_num = 1
        
        while len(winners_bracket) > 1 or len(losers_bracket) > 1:
            # Winners bracket
            new_winners = []
            for i in range(0, len(winners_bracket), 2):
                if i + 1 >= len(winners_bracket):
                    new_winners.append(winners_bracket[i])
                    continue
                
                p_a = winners_bracket[i]
                p_b = winners_bracket[i + 1]
                
                match = self._run_match(p_a, p_b, challenge, round_num)
                
                if match.winner == p_a.participant_id:
                    new_winners.append(p_a)
                    losers_bracket.append(p_b)
                else:
                    new_winners.append(p_b)
                    losers_bracket.append(p_a)
            
            winners_bracket = new_winners
            
            # Losers bracket
            new_losers = []
            for i in range(0, len(losers_bracket), 2):
                if i + 1 >= len(losers_bracket):
                    new_losers.append(losers_bracket[i])
                    continue
                
                p_a = losers_bracket[i]
                p_b = losers_bracket[i + 1]
                
                match = self._run_match(p_a, p_b, challenge, round_num)
                
                if match.winner == p_a.participant_id:
                    new_losers.append(p_a)
                    p_b.eliminated = True
                else:
                    new_losers.append(p_b)
                    p_a.eliminated = True
            
            losers_bracket = new_losers
            round_num += 1
        
        # Grand finals
        if winners_bracket and losers_bracket:
            match = self._run_match(
                winners_bracket[0],
                losers_bracket[0],
                challenge,
                round_num
            )
            return match.winner
        
        return winners_bracket[0].participant_id if winners_bracket else None
    
    def _run_swiss(self, challenge: str, rounds: int = 5) -> Optional[str]:
        """Run Swiss system tournament."""
        participants = list(self._participants.values())
        
        for round_num in range(1, rounds + 1):
            # Sort by wins
            participants.sort(key=lambda p: -p.wins)
            
            # Pair adjacent players
            paired = set()
            for i, p_a in enumerate(participants):
                if p_a.participant_id in paired:
                    continue
                
                for p_b in participants[i+1:]:
                    if p_b.participant_id in paired:
                        continue
                    
                    self._run_match(p_a, p_b, challenge, round_num)
                    paired.add(p_a.participant_id)
                    paired.add(p_b.participant_id)
                    break
        
        # Winner has most wins
        winner = max(participants, key=lambda p: (p.wins, -p.losses))
        return winner.participant_id
    
    def _run_match(
        self,
        p_a: Participant,
        p_b: Participant,
        challenge: str,
        round_num: int
    ) -> Match:
        """Run a single match."""
        self._emit("match_started", {
            "a": p_a.participant_id,
            "b": p_b.participant_id,
            "round": round_num
        })
        
        match = self.judge.judge_match(p_a, p_b, challenge)
        match.round_num = round_num
        
        # Update stats
        if match.winner == p_a.participant_id:
            p_a.wins += 1
            p_b.losses += 1
            elo_a, elo_b = EloSystem.update_ratings(
                p_a.elo_rating, p_b.elo_rating, 1.0, 0.0
            )
        elif match.winner == p_b.participant_id:
            p_b.wins += 1
            p_a.losses += 1
            elo_a, elo_b = EloSystem.update_ratings(
                p_a.elo_rating, p_b.elo_rating, 0.0, 1.0
            )
        else:
            p_a.draws += 1
            p_b.draws += 1
            elo_a, elo_b = EloSystem.update_ratings(
                p_a.elo_rating, p_b.elo_rating, 0.5, 0.5
            )
        
        p_a.elo_rating = elo_a
        p_b.elo_rating = elo_b
        
        self._matches.append(match)
        
        self._emit("match_ended", {
            "winner": match.winner,
            "score_a": match.score_a,
            "score_b": match.score_b
        })
        
        return match
    
    def add_listener(self, callback: Callable[[str, Any], None]):
        """Add event listener."""
        self._listeners.append(callback)
    
    def _emit(self, event: str, data: Any):
        """Emit event."""
        for listener in self._listeners:
            try:
                listener(event, data)
            except Exception as e:
                logger.error(f"Listener error: {e}")
    
    def get_bracket_text(self, result: TournamentResult) -> str:
        """Generate text bracket visualization."""
        lines = [
            f"Tournament: {result.tournament_type.value}",
            f"Challenge: {result.challenge}",
            "=" * 50,
            ""
        ]
        
        # Matches by round
        matches_by_round: Dict[int, List[Match]] = {}
        for match in result.matches:
            if match.round_num not in matches_by_round:
                matches_by_round[match.round_num] = []
            matches_by_round[match.round_num].append(match)
        
        for round_num in sorted(matches_by_round.keys()):
            lines.append(f"Round {round_num}:")
            for match in matches_by_round[round_num]:
                winner_marker = lambda p: "*" if p == match.winner else " "
                lines.append(
                    f"  {winner_marker(match.participant_a)}{match.participant_a} ({match.score_a:.1f}) "
                    f"vs {winner_marker(match.participant_b)}{match.participant_b} ({match.score_b:.1f})"
                )
            lines.append("")
        
        lines.append(f"Champion: {result.champion}")
        lines.append("")
        lines.append("Final Standings:")
        for i, (p_id, wins) in enumerate(result.final_standings, 1):
            elo_change = result.elo_changes.get(p_id, 0)
            sign = "+" if elo_change >= 0 else ""
            lines.append(f"  {i}. {p_id}: {wins} wins (ELO {sign}{elo_change:.0f})")
        
        return "\n".join(lines)


# Convenience functions
def quick_tournament(
    agents: Dict[str, Any],
    challenge: str,
    tournament_type: TournamentType = TournamentType.SINGLE_ELIMINATION
) -> TournamentResult:
    """
    Quickly run a tournament.
    
    Args:
        agents: Dict of agent_id -> agent
        challenge: The challenge
        tournament_type: Tournament format
        
    Returns:
        Tournament result
    """
    tournament = Tournament(tournament_type)
    
    for agent_id, agent in agents.items():
        tournament.add_participant(agent_id, agent)
    
    return tournament.run(challenge)
