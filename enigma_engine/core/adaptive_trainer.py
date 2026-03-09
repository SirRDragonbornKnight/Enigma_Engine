"""
Enigma Engine - Adaptive Training Pipeline
=============================================

Combines:
- TC-C3: Continuous adaptive loop — TRAINER evaluates STUDENT,
  adjusts difficulty, generates lessons based on current ability.
- SA-B: Auto-chain stages — runs BASICS → CONVERSATION → COMMANDS → WEB
  sequentially, with Phase 3 tests deciding advance/retry.
- SA-C: Training plan — saveable as JSON, resume on crash, run overnight.

The TrainingPlan tracks progress across stages. The AdaptiveProbe
evaluates STUDENT ability. Together they let the training loop
run autonomously from start to finish, adapting to the model's
real skill level.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# All stages in learning order
ALL_STAGES = ["basics", "conversation", "commands", "web"]

# Difficulty levels within each stage (TC-C3)
DIFFICULTY_LEVELS = ["simple", "medium", "advanced"]


@dataclass
class StageResult:
    """Results from one attempt at a training stage."""
    stage: str
    attempt: int
    difficulty: str = "simple"
    scores: list[float] = field(default_factory=list)
    avg_score: float = 0.0
    status: str = "pending"  # pending, passed, failed, skipped
    epochs_trained: int = 0
    pairs_generated: int = 0
    best_loss: float = float("inf")
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageResult:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class TrainingPlan:
    """Persistent training plan — saves/loads as JSON for resume.

    Tracks the full auto-chain pipeline: which stages have been
    completed, current progress, scores, and adaptive difficulty.
    """
    # Identity
    student_path: str = ""
    trainer_path: str = ""
    student_name: str = ""
    trainer_name: str = ""

    # Stage progression
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    current_stage_idx: int = 0

    # Training params
    epochs_per_stage: int = 10
    pairs_per_stage: int = 20
    learning_rate: float = 5e-5
    max_retries: int = 3

    # Adaptive mode (TC-C3)
    adaptive: bool = True
    current_difficulty: str = "simple"

    # Results tracking
    stage_results: list[dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    last_updated: str = ""
    completed_at: str = ""

    # State
    status: str = "pending"  # pending, running, paused, completed, failed

    # Training brief and focus (carried from GUI)
    training_brief: str = ""
    focus_field: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def current_stage(self) -> str | None:
        """Current stage name, or None if all done."""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """True if all stages passed or plan is done."""
        return (self.status == "completed"
                or self.current_stage_idx >= len(self.stages))

    @property
    def current_attempt(self) -> int:
        """How many times we've tried the current stage."""
        stage = self.current_stage
        if stage is None:
            return 0
        return sum(1 for r in self.stage_results
                   if r.get("stage") == stage)

    def decide_action(self, avg_score: float) -> str:
        """Decide what to do after a Phase 3 test.

        Always advances to the next stage. Score thresholds are not
        used — the model progresses through the full stage pipeline
        regardless.

        Args:
            avg_score: Average test score (1-10). Stored for logging
                but does not gate progression.

        Returns:
            One of: "advance", "complete"
        """
        if self.current_stage_idx + 1 >= len(self.stages):
            return "complete"
        return "advance"

    def advance_stage(self) -> bool:
        """Move to the next stage.

        Returns:
            True if there's a next stage, False if complete.
        """
        self.current_stage_idx += 1
        # Reset difficulty for the new stage
        self.current_difficulty = "simple"
        if self.current_stage_idx >= len(self.stages):
            self.status = "completed"
            self.completed_at = datetime.now().isoformat()
            return False
        return True

    def record_result(self, result: StageResult) -> None:
        """Record a stage attempt result."""
        self.stage_results.append(result.to_dict())
        self.last_updated = datetime.now().isoformat()

    def save(self, path: str | Path) -> None:
        """Save plan to JSON file."""
        from enigma_engine.core.safe_save import atomic_write_text
        path = Path(path)
        data = asdict(self)
        atomic_write_text(path, json.dumps(data, indent=2, default=str))
        logger.info("Training plan saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TrainingPlan:
        """Load plan from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Filter unknown keys for forward compat
        known = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in known}
        plan = cls(**filtered)
        logger.info("Training plan loaded: %s (stage %d/%d)",
                     path, plan.current_stage_idx + 1,
                     len(plan.stages))
        return plan

    def summary(self) -> str:
        """Human-readable summary of plan state."""
        lines = [
            f"Training Plan: {self.student_name} ← {self.trainer_name}",
            f"Status: {self.status.upper()}",
            f"Stage: {self.current_stage or 'DONE'} "
            f"({self.current_stage_idx + 1}/{len(self.stages)})",
        ]
        if self.adaptive:
            lines.append(f"Difficulty: {self.current_difficulty}")
        for r in self.stage_results:
            s = r.get("stage", "?")
            a = r.get("attempt", 0)
            avg = r.get("avg_score", 0)
            st = r.get("status", "?")
            lines.append(f"  {s} (attempt {a}): {avg:.1f}/10 [{st}]")
        return "\n".join(lines)


def build_adaptive_prompt(
    index: int,
    total: int,
    stage: str,
    difficulty: str,
) -> str:
    """Build a generation prompt adapted to the current difficulty.

    Extends _build_generation_prompt with difficulty awareness.
    At 'simple' level, generates ultra-basic content (alphabet,
    single words). At 'advanced', generates complex content.

    Args:
        index: 1-based example index.
        total: Total examples to generate.
        stage: Training stage name.
        difficulty: One of "simple", "medium", "advanced".

    Returns:
        A prompt string for the teacher model.
    """
    # Difficulty modifiers
    diff_hints = {
        "simple": {
            "basics": (
                "Generate VERY simple content:\n"
                "- Single common words (cat, dog, run, big)\n"
                "- Two-word phrases (hello world, good morning)\n"
                "- Ultra-short sentences (The cat sat.)\n"
                "- Basic letter/number patterns (A B C, 1 2 3)\n"
                "The student barely understands language. "
                "Keep it as simple as possible."
            ),
            "conversation": (
                "Generate a VERY simple 2-turn conversation:\n"
                "User: <one simple question>\n"
                "AI: <one short answer>\n"
                "Use everyday words. One sentence per turn max."
            ),
            "commands": (
                "Generate a simple command example:\n"
                "Q: <basic question>\n"
                "A: <short answer with one [CMD]...[/CMD]>\n"
                "Keep the command as simple as possible."
            ),
            "web": (
                "Generate a simple web search example:\n"
                "Q: <basic factual question>\n"
                "A: <short answer using [CMD]search.web ...[/CMD]>\n"
                "Keep it straightforward."
            ),
        },
        "medium": {
            "basics": (
                "Generate a clear, well-formed example:\n"
                "- A factual statement (2-3 sentences)\n"
                "- A definition with an example\n"
                "- A short opinion with reasoning\n"
                "Use varied vocabulary. Be natural."
            ),
            "conversation": (
                "Generate a natural 3-4 turn conversation:\n"
                "User: <question or statement>\n"
                "AI: <thoughtful response>\n"
                "User: <follow-up>\n"
                "AI: <response>\n"
                "Show good conversational flow."
            ),
            "commands": (
                "Generate a command example with context:\n"
                "Q: <practical question>\n"
                "A: <explanation + [CMD]...[/CMD] + result>\n"
                "Show the AI reasoning about which command to use."
            ),
            "web": (
                "Generate a web research example:\n"
                "Q: <question needing current info>\n"
                "A: <answer using [CMD]search.web ...[/CMD] "
                "with source citation>\n"
                "Show the AI explaining what it found."
            ),
        },
        "advanced": {
            "basics": (
                "Generate sophisticated content:\n"
                "- A nuanced explanation of a concept\n"
                "- An analogy that illuminates a complex idea\n"
                "- A short paragraph with varied sentence structure\n"
                "Use rich vocabulary. Show depth."
            ),
            "conversation": (
                "Generate a complex multi-turn dialogue:\n"
                "- 4+ turns with topic evolution\n"
                "- Show the AI asking clarifying questions\n"
                "- Include a correction or refinement\n"
                "- Vary tone: some playful, some serious\n"
                "Make it feel like a real deep conversation."
            ),
            "commands": (
                "Generate an advanced command example:\n"
                "- Multi-step task requiring 2+ commands\n"
                "- Show reasoning about approach\n"
                "- Handle potential errors or edge cases\n"
                "The AI should demonstrate real problem-solving."
            ),
            "web": (
                "Generate an advanced research example:\n"
                "- Complex question needing multiple searches\n"
                "- Cross-reference multiple sources\n"
                "- Synthesize findings into a coherent answer\n"
                "Show deep research methodology."
            ),
        },
    }

    hints = diff_hints.get(difficulty, diff_hints["medium"])
    stage_hint = hints.get(stage, hints.get("basics", ""))

    return (
        f"Training example #{index} of {total}.\n"
        f"Stage: {stage.upper()} | Difficulty: {difficulty.upper()}\n\n"
        f"{stage_hint}\n\n"
        "Write ONLY the example, no labels or meta-text. "
        "Make it different from previous examples."
    )
