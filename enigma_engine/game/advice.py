"""
Real-time Game Advice System

Proactive AI advice system for gaming.
Monitors game state and provides contextual tips, strategies, and alerts.

FILE: enigma_engine/game/advice.py
TYPE: Game
MAIN CLASSES: GameAdvisor, AdviceEngine, ContextAnalyzer
"""

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdviceCategory(Enum):
    """Categories of game advice."""
    STRATEGY = "strategy"
    ITEM_BUILD = "item_build"
    MECHANICS = "mechanics"
    MAP_AWARENESS = "map_awareness"
    ECONOMY = "economy"
    TEAM_PLAY = "team_play"
    TIMING = "timing"
    COUNTER_PLAY = "counter_play"
    SAFETY = "safety"
    GENERAL_TIP = "general_tip"


class AdvicePriority(Enum):
    """Priority levels for advice."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class AdviceTrigger(Enum):
    """What triggers advice."""
    PERIODIC = "periodic"
    EVENT = "event"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    USER_REQUEST = "user_request"


@dataclass
class Advice:
    """A piece of game advice."""
    id: str
    category: AdviceCategory
    priority: AdvicePriority
    title: str
    content: str
    trigger: AdviceTrigger
    game_id: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    shown: bool = False
    dismissed: bool = False


@dataclass
class GameContext:
    """Current game state context."""
    game_id: str
    game_state: str = "unknown"  # menu, loading, playing, paused
    game_time_seconds: float = 0
    player_health: Optional[float] = None
    player_resources: dict[str, float] = field(default_factory=dict)
    nearby_enemies: int = 0
    current_objective: Optional[str] = None
    recent_events: list[str] = field(default_factory=list)
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdviceRule:
    """Rule for generating advice."""
    rule_id: str
    game_id: Optional[str]  # None = all games
    category: AdviceCategory
    trigger_type: AdviceTrigger
    condition: Callable[[GameContext], bool]
    generator: Callable[[GameContext], str]
    priority: AdvicePriority = AdvicePriority.MEDIUM
    cooldown_seconds: float = 60
    last_triggered: float = 0


class ContextAnalyzer:
    """
    Analyze game context to determine optimal advice.
    """
    
    def __init__(self):
        self._context_history: list[GameContext] = []
        self._max_history = 100
    
    def add_context(self, context: GameContext):
        """Add context snapshot to history."""
        self._context_history.append(context)
        if len(self._context_history) > self._max_history:
            self._context_history.pop(0)
    
    def analyze_trends(self) -> dict[str, Any]:
        """Analyze context trends."""
        if len(self._context_history) < 5:
            return {}
        
        recent = self._context_history[-10:]
        
        trends = {
            "health_trend": self._calculate_trend([c.player_health for c in recent if c.player_health]),
            "enemy_trend": self._calculate_trend([c.nearby_enemies for c in recent]),
            "time_in_game": recent[-1].game_time_seconds if recent else 0
        }
        
        return trends
    
    def _calculate_trend(self, values: list[float]) -> str:
        """Calculate trend from values."""
        if len(values) < 2:
            return "stable"
        
        diff = values[-1] - values[0]
        avg = sum(values) / len(values)
        
        if avg == 0:
            return "stable"
        
        change_percent = abs(diff) / avg * 100
        
        if change_percent < 10:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def detect_patterns(self) -> list[str]:
        """Detect patterns in recent gameplay."""
        patterns = []
        
        if len(self._context_history) < 10:
            return patterns
        
        recent = self._context_history[-20:]
        
        # Detect repeated deaths (health going to 0)
        zero_health_count = sum(1 for c in recent if c.player_health == 0)
        if zero_health_count >= 3:
            patterns.append("repeated_deaths")
        
        # Detect low resource management
        low_resource_count = sum(
            1 for c in recent 
            for v in c.player_resources.values() 
            if v < 10
        )
        if low_resource_count >= 5:
            patterns.append("low_resources")
        
        # Detect high enemy encounters
        high_enemy_count = sum(1 for c in recent if c.nearby_enemies >= 3)
        if high_enemy_count >= 5:
            patterns.append("frequent_combat")
        
        return patterns


class AdviceEngine:
    """
    Engine for generating and managing advice.
    """
    
    def __init__(self):
        self._rules: list[AdviceRule] = []
        self._pending_advice: queue.PriorityQueue = queue.PriorityQueue()
        self._advice_history: list[Advice] = []
        self._analyzer = ContextAnalyzer()
        
        # Default rules
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default advice rules."""
        # Low health warning
        self.add_rule(AdviceRule(
            rule_id="low_health",
            game_id=None,
            category=AdviceCategory.SAFETY,
            trigger_type=AdviceTrigger.THRESHOLD,
            condition=lambda ctx: ctx.player_health is not None and ctx.player_health < 25,
            generator=lambda ctx: f"Health critical at {ctx.player_health:.0f}%! Consider retreating or using healing items.",
            priority=AdvicePriority.URGENT,
            cooldown_seconds=30
        ))
        
        # Early game tips
        self.add_rule(AdviceRule(
            rule_id="early_game",
            game_id=None,
            category=AdviceCategory.STRATEGY,
            trigger_type=AdviceTrigger.THRESHOLD,
            condition=lambda ctx: 60 < ctx.game_time_seconds < 180 and ctx.game_state == "playing",
            generator=lambda ctx: "Early game tip: Focus on securing resources and establishing map control.",
            priority=AdvicePriority.LOW,
            cooldown_seconds=120
        ))
        
        # Enemy encounter
        self.add_rule(AdviceRule(
            rule_id="enemy_nearby",
            game_id=None,
            category=AdviceCategory.MAP_AWARENESS,
            trigger_type=AdviceTrigger.THRESHOLD,
            condition=lambda ctx: ctx.nearby_enemies >= 3,
            generator=lambda ctx: f"Multiple enemies detected ({ctx.nearby_enemies})! Coordinate with team or find cover.",
            priority=AdvicePriority.HIGH,
            cooldown_seconds=15
        ))
    
    def add_rule(self, rule: AdviceRule):
        """Add an advice rule."""
        self._rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a rule by ID."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]
    
    def process_context(self, context: GameContext) -> list[Advice]:
        """
        Process game context and generate advice.
        
        Args:
            context: Current game context
        
        Returns:
            List of generated advice
        """
        self._analyzer.add_context(context)
        generated = []
        current_time = time.time()
        
        for rule in self._rules:
            # Check cooldown
            if current_time - rule.last_triggered < rule.cooldown_seconds:
                continue
            
            # Check game ID match
            if rule.game_id and rule.game_id != context.game_id:
                continue
            
            # Check condition
            try:
                if rule.condition(context):
                    advice_content = rule.generator(context)
                    
                    advice = Advice(
                        id=f"{rule.rule_id}_{int(current_time)}",
                        category=rule.category,
                        priority=rule.priority,
                        title=rule.category.value.replace('_', ' ').title(),
                        content=advice_content,
                        trigger=rule.trigger_type,
                        game_id=context.game_id,
                        context={"game_time": context.game_time_seconds}
                    )
                    
                    generated.append(advice)
                    rule.last_triggered = current_time
                    
            except Exception as e:
                logger.debug(f"Error evaluating rule {rule.rule_id}: {e}")
        
        return generated
    
    def get_contextual_tips(
        self,
        context: GameContext,
        category: Optional[AdviceCategory] = None,
        count: int = 3
    ) -> list[str]:
        """
        Get contextual tips based on game state.
        
        Args:
            context: Current game context
            category: Filter by category
            count: Number of tips
        
        Returns:
            List of tip strings
        """
        trends = self._analyzer.analyze_trends()
        patterns = self._analyzer.detect_patterns()
        
        tips = []
        
        # Pattern-based tips
        if "repeated_deaths" in patterns:
            tips.append("Consider adjusting your approach - try observing enemy patterns before engaging.")
        
        if "low_resources" in patterns:
            tips.append("Resource management tip: Prioritize farms or resource nodes.")
        
        if "frequent_combat" in patterns:
            tips.append("Lots of combat! Make sure to manage cooldowns and health between fights.")
        
        # Trend-based tips
        if trends.get("health_trend") == "decreasing":
            tips.append("Your health has been trending down - consider safer positioning.")
        
        # Time-based tips
        game_time = context.game_time_seconds
        if game_time < 300:
            tips.append("Early game: Focus on economy and map control.")
        elif game_time < 900:
            tips.append("Mid game: Look for opportunities to secure objectives.")
        else:
            tips.append("Late game: Team fights and objective control are crucial.")
        
        # Shuffle and limit
        random.shuffle(tips)
        return tips[:count]


class GameAdvisor:
    """
    Main advisor interface for games.
    Manages advice generation, timing, and delivery.
    """
    
    def __init__(
        self,
        ai_model: Any = None,
        advice_callback: Callable[[Advice], None] = None
    ):
        self._engine = AdviceEngine()
        self._ai_model = ai_model
        self._callback = advice_callback
        
        self._current_game: Optional[str] = None
        self._context: Optional[GameContext] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._advice_queue: queue.Queue = queue.Queue()
        self._periodic_interval = 60  # seconds
        self._last_periodic = 0
    
    def start(self, game_id: str):
        """Start the advisor for a game."""
        self._current_game = game_id
        self._context = GameContext(game_id=game_id)
        self._running = True
        
        self._thread = threading.Thread(target=self._advice_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Game advisor started for {game_id}")
    
    def stop(self):
        """Stop the advisor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Game advisor stopped")
    
    def update_context(self, **kwargs):
        """Update game context with new data."""
        if not self._context:
            return
        
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
            else:
                self._context.custom_data[key] = value
    
    def _advice_loop(self):
        """Background loop for advice generation."""
        while self._running:
            try:
                if self._context:
                    # Process rules
                    advice_list = self._engine.process_context(self._context)
                    
                    for advice in advice_list:
                        self._deliver_advice(advice)
                    
                    # Periodic tips
                    current_time = time.time()
                    if current_time - self._last_periodic > self._periodic_interval:
                        self._generate_periodic_tip()
                        self._last_periodic = current_time
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in advice loop: {e}")
                time.sleep(5)
    
    def _deliver_advice(self, advice: Advice):
        """Deliver advice through callback."""
        if self._callback:
            try:
                self._callback(advice)
            except Exception as e:
                logger.error(f"Error delivering advice: {e}")
        
        self._advice_queue.put(advice)
    
    def _generate_periodic_tip(self):
        """Generate periodic tip using AI model if available."""
        if not self._context:
            return
        
        tips = self._engine.get_contextual_tips(self._context, count=1)
        
        if tips:
            advice = Advice(
                id=f"periodic_{int(time.time())}",
                category=AdviceCategory.GENERAL_TIP,
                priority=AdvicePriority.LOW,
                title="Tip",
                content=tips[0],
                trigger=AdviceTrigger.PERIODIC,
                game_id=self._current_game
            )
            self._deliver_advice(advice)
    
    def ask_for_advice(self, question: str) -> str:
        """
        Ask AI for specific advice.
        
        Args:
            question: User's question
        
        Returns:
            AI response
        """
        if self._ai_model and hasattr(self._ai_model, 'generate'):
            # Build context prompt
            context_str = ""
            if self._context:
                context_str = f"\nCurrent game state: {self._context.game_state}"
                if self._context.player_health:
                    context_str += f", Health: {self._context.player_health}%"
                if self._context.game_time_seconds:
                    mins = int(self._context.game_time_seconds // 60)
                    context_str += f", Game time: {mins}m"
            
            prompt = f"Game: {self._current_game}{context_str}\n\nQuestion: {question}\n\nAdvice:"
            
            try:
                response = self._ai_model.generate(prompt, max_tokens=150)
                return response
            except Exception as e:
                logger.error(f"Error generating AI advice: {e}")
        
        # Fallback to tips
        tips = self._engine.get_contextual_tips(self._context or GameContext(game_id="unknown"))
        return tips[0] if tips else "No advice available at the moment."
    
    def get_pending_advice(self) -> list[Advice]:
        """Get all pending advice."""
        advice_list = []
        while not self._advice_queue.empty():
            try:
                advice_list.append(self._advice_queue.get_nowait())
            except queue.Empty:
                break
        return advice_list
    
    def add_custom_rule(
        self,
        rule_id: str,
        category: AdviceCategory,
        condition: Callable[[GameContext], bool],
        generator: Callable[[GameContext], str],
        priority: AdvicePriority = AdvicePriority.MEDIUM,
        cooldown: float = 60
    ):
        """Add a custom advice rule."""
        rule = AdviceRule(
            rule_id=rule_id,
            game_id=self._current_game,
            category=category,
            trigger_type=AdviceTrigger.PATTERN,
            condition=condition,
            generator=generator,
            priority=priority,
            cooldown_seconds=cooldown
        )
        self._engine.add_rule(rule)
    
    def set_periodic_interval(self, seconds: int):
        """Set interval for periodic tips."""
        self._periodic_interval = max(10, seconds)
    
    def set_advice_callback(self, callback: Callable[[Advice], None]):
        """Set callback for advice delivery."""
        self._callback = callback


def create_game_advisor(
    game_id: str,
    ai_model: Any = None,
    callback: Callable[[Advice], None] = None
) -> GameAdvisor:
    """
    Create and start a game advisor.
    
    Args:
        game_id: Game identifier
        ai_model: AI model for generating advice
        callback: Function to call when advice is generated
    
    Returns:
        Started GameAdvisor instance
    """
    advisor = GameAdvisor(ai_model, callback)
    advisor.start(game_id)
    return advisor
