"""
User-Teachable Behavior Preferences System for Enigma AI Engine

Allows users to teach the AI custom behavior rules through natural conversation.
For example: "Whenever you teleport, spawn a portal gun first"

This system:
1. Detects behavior preference statements in conversation
2. Stores them persistently
3. Applies them during tool execution

Usage:
    from enigma_engine.learning.behavior_preferences import BehaviorManager, get_behavior_manager
    
    # Get singleton manager
    manager = get_behavior_manager()
    
    # User says: "whenever you teleport, spawn a portal gun first"
    manager.learn_from_statement("whenever you teleport, spawn a portal gun first")
    
    # When AI executes teleport tool, manager suggests:
    actions = manager.get_actions_for("teleport")
    # Returns: [BehaviorAction(timing='before', tool='spawn_object', params={'object': 'portal_gun'})]
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActionTiming(Enum):
    """When to execute the learned behavior relative to the trigger action."""
    BEFORE = "before"      # Run before the trigger action
    AFTER = "after"        # Run after the trigger action
    INSTEAD = "instead"    # Replace the trigger action entirely
    WITH = "with"          # Run alongside (both execute)


@dataclass
class BehaviorAction:
    """An action to take as part of a behavior rule."""
    timing: ActionTiming
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class BehaviorRule:
    """A learned behavior rule from user teaching."""
    id: str
    trigger_action: str           # Tool/action that triggers this rule (e.g., "teleport")
    trigger_keywords: List[str]   # Alternative triggers (e.g., ["teleport", "warp", "jump"])
    actions: List[BehaviorAction] # What to do when triggered
    condition: Optional[str]      # Optional condition (e.g., "if near enemy")
    original_statement: str       # The user's original teaching statement
    created_at: float = field(default_factory=time.time)
    enabled: bool = True
    use_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trigger_action": self.trigger_action,
            "trigger_keywords": self.trigger_keywords,
            "actions": [
                {
                    "timing": a.timing.value,
                    "tool_name": a.tool_name,
                    "params": a.params,
                    "description": a.description,
                }
                for a in self.actions
            ],
            "condition": self.condition,
            "original_statement": self.original_statement,
            "created_at": self.created_at,
            "enabled": self.enabled,
            "use_count": self.use_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorRule":
        """Create from dictionary."""
        actions = [
            BehaviorAction(
                timing=ActionTiming(a["timing"]),
                tool_name=a["tool_name"],
                params=a.get("params", {}),
                description=a.get("description", ""),
            )
            for a in data.get("actions", [])
        ]
        return cls(
            id=data["id"],
            trigger_action=data["trigger_action"],
            trigger_keywords=data.get("trigger_keywords", []),
            actions=actions,
            condition=data.get("condition"),
            original_statement=data.get("original_statement", ""),
            created_at=data.get("created_at", time.time()),
            enabled=data.get("enabled", True),
            use_count=data.get("use_count", 0),
        )


class BehaviorManager:
    """
    Manages user-taught behavior preferences.
    
    Detects behavior rules from user statements, stores them,
    and provides them during tool execution.
    """
    
    # Patterns to detect behavior teaching statements
    # Each tuple: (pattern, timing, capture_groups_meaning)
    BEHAVIOR_PATTERNS: List[Tuple[str, ActionTiming, str]] = [
        # "whenever you X, spawn/use/do Y first" - capture full action phrase
        (r"whenever you (?:do |perform )?([^,]+),\s*((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)(?:\s*first)?", 
         ActionTiming.BEFORE, "trigger,action"),
        
        # "before you X, always spawn/use/do Y"
        (r"before you (?:do |perform )?([^,]+),\s*(?:always\s+)?((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)", 
         ActionTiming.BEFORE, "trigger,action"),
        
        # "when you X, spawn/use/do Y first"
        (r"when(?:ever)? you (?:do |perform )?([^,]+),\s*((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)(?:\s*first)?", 
         ActionTiming.BEFORE, "trigger,action"),
        
        # "after you X, always spawn/use/do Y"
        (r"after you (?:do |perform )?([^,]+),\s*(?:always\s+)?((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)", 
         ActionTiming.AFTER, "trigger,action"),
        
        # "when you X, also spawn/use Y"
        (r"when(?:ever)? you (?:do |perform )?([^,]+),\s*(?:also\s+)?((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)", 
         ActionTiming.WITH, "trigger,action"),
        
        # "instead of X, spawn/use/do Y"
        (r"instead of (?:doing )?([^,]+),\s*((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+)", 
         ActionTiming.INSTEAD, "trigger,action"),
        
        # "always spawn X before Y"
        (r"always ((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+?)\s*before (?:you )?(?:do |perform )?(.+)", 
         ActionTiming.BEFORE, "action,trigger"),
        
        # "remember to spawn X whenever you Y"
        (r"remember to ((?:spawn|use|create|summon|do|perform|get|grab|cast|hold)\s+(?:a\s+)?[^.!?]+?)\s*whenever you (?:do |perform )?(.+)", 
         ActionTiming.BEFORE, "action,trigger"),
        
        # "for X, use Y" / "when doing X, use Y"
        (r"(?:for|when doing|during) ([^,]+),\s*(?:always\s+)?use (.+)", 
         ActionTiming.WITH, "trigger,action"),
    ]
    
    # Known action/tool mappings for natural language
    ACTION_MAPPINGS: Dict[str, str] = {
        # Teleportation
        "teleport": "teleport",
        "warp": "teleport",
        "jump to": "teleport",
        "go to": "move_to",
        
        # Objects
        "spawn": "spawn_object",
        "create": "spawn_object",
        "summon": "spawn_object",
        "get": "spawn_object",
        "grab": "hold_item",
        "hold": "hold_item",
        "drop": "drop_item",
        "throw": "throw_item",
        
        # Movement
        "walk": "walk_to",
        "run": "run_to",
        "move": "move_to",
        "sit": "sit_down",
        "stand": "stand_up",
        "jump": "jump",
        "dance": "dance",
        
        # Interaction
        "wave": "wave",
        "point": "point_at",
        "look": "look_at",
        "eat": "eat",
        "drink": "drink",
        
        # Combat/Game
        "attack": "attack",
        "defend": "defend",
        "block": "block",
        "shoot": "shoot",
        "cast": "cast_spell",
    }
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize behavior manager.
        
        Args:
            storage_dir: Directory to store behavior rules
        """
        self.storage_dir = storage_dir or Path("memory/behaviors")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self._rules: Dict[str, BehaviorRule] = {}
        self._trigger_index: Dict[str, List[str]] = {}  # trigger -> rule_ids
        
        # Compile patterns
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), timing, groups)
            for p, timing, groups in self.BEHAVIOR_PATTERNS
        ]
        
        # Load existing rules
        self._load_rules()
        
        logger.info(f"BehaviorManager initialized with {len(self._rules)} rules")
    
    def learn_from_statement(self, statement: str, user_id: str = "default") -> Optional[BehaviorRule]:
        """
        Try to learn a behavior rule from a user statement.
        
        Args:
            statement: User's natural language statement
            user_id: ID of the user teaching the behavior
            
        Returns:
            BehaviorRule if successfully parsed, None otherwise
        """
        statement = statement.strip()
        
        for pattern, timing, groups_meaning in self._compiled_patterns:
            match = pattern.search(statement)
            if match:
                try:
                    return self._create_rule_from_match(
                        match, timing, groups_meaning, statement
                    )
                except Exception as e:
                    logger.debug(f"Failed to create rule from match: {e}")
                    continue
        
        logger.debug(f"No behavior pattern matched: {statement}")
        return None
    
    def _create_rule_from_match(
        self, 
        match: re.Match, 
        timing: ActionTiming,
        groups_meaning: str,
        original: str
    ) -> BehaviorRule:
        """Create a behavior rule from a regex match."""
        groups = match.groups()
        meanings = groups_meaning.split(",")
        
        trigger_text = ""
        action_text = ""
        
        for i, meaning in enumerate(meanings):
            if i < len(groups):
                if meaning == "trigger":
                    trigger_text = groups[i].strip()
                elif meaning == "action":
                    action_text = groups[i].strip()
        
        # Map natural language to tool names
        trigger_tool = self._map_to_tool(trigger_text)
        action_tool, action_params = self._parse_action(action_text)
        
        # Generate unique ID
        rule_id = f"rule_{int(time.time() * 1000)}_{len(self._rules)}"
        
        # Create the rule
        rule = BehaviorRule(
            id=rule_id,
            trigger_action=trigger_tool,
            trigger_keywords=self._extract_keywords(trigger_text),
            actions=[
                BehaviorAction(
                    timing=timing,
                    tool_name=action_tool,
                    params=action_params,
                    description=action_text,
                )
            ],
            condition=None,
            original_statement=original,
        )
        
        # Store the rule
        self._add_rule(rule)
        
        logger.info(f"Learned behavior: {timing.value} '{trigger_tool}' -> '{action_tool}'")
        return rule
    
    def _map_to_tool(self, text: str) -> str:
        """Map natural language action to tool name."""
        text_lower = text.lower().strip()
        
        # Check direct mappings first
        for phrase, tool in self.ACTION_MAPPINGS.items():
            if phrase in text_lower:
                return tool
        
        # Extract first verb as fallback
        words = text_lower.split()
        if words:
            first_word = words[0]
            if first_word in self.ACTION_MAPPINGS:
                return self.ACTION_MAPPINGS[first_word]
            return first_word  # Use the verb itself as tool name
        
        return text_lower
    
    def _parse_action(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Parse action text into tool name and parameters."""
        text_lower = text.lower().strip()
        params: Dict[str, Any] = {}
        
        # Remove trailing filler words
        text_lower = re.sub(r"\s*(first|too|also|as well)\s*$", "", text_lower)
        
        # Check for "spawn X" / "use X" / "create X" patterns
        spawn_match = re.search(r"(?:spawn|create|summon|use|get|grab)\s+(?:a\s+)?(?:the\s+)?(.+)", text_lower)
        if spawn_match:
            object_name = spawn_match.group(1).strip()
            # Remove trailing punctuation
            object_name = re.sub(r"[.!?]+$", "", object_name)
            return "spawn_object", {"object": object_name}
        
        # Check for "hold X"
        hold_match = re.search(r"hold\s+(?:a\s+)?(?:the\s+)?(.+)", text_lower)
        if hold_match:
            item = hold_match.group(1).strip()
            item = re.sub(r"[.!?]+$", "", item)
            return "hold_item", {"item": item}
        
        # Check for movement actions
        move_match = re.search(r"(?:go|move|walk|run)\s+to\s+(.+)", text_lower)
        if move_match:
            destination = move_match.group(1).strip()
            return "move_to", {"destination": destination}
        
        # Fallback: use the first word as tool name
        tool = self._map_to_tool(text)
        return tool, params
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for trigger matching."""
        text_lower = text.lower()
        keywords = []
        
        # Add the main action
        for phrase, tool in self.ACTION_MAPPINGS.items():
            if phrase in text_lower:
                keywords.append(phrase)
                keywords.append(tool)
        
        # Add individual words
        words = re.findall(r"\b\w+\b", text_lower)
        for word in words:
            if len(word) > 2 and word not in ["the", "and", "you", "for"]:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _add_rule(self, rule: BehaviorRule):
        """Add a rule and update indices."""
        self._rules[rule.id] = rule
        
        # Index by trigger
        trigger = rule.trigger_action.lower()
        if trigger not in self._trigger_index:
            self._trigger_index[trigger] = []
        self._trigger_index[trigger].append(rule.id)
        
        # Index by keywords
        for keyword in rule.trigger_keywords:
            keyword = keyword.lower()
            if keyword not in self._trigger_index:
                self._trigger_index[keyword] = []
            if rule.id not in self._trigger_index[keyword]:
                self._trigger_index[keyword].append(rule.id)
        
        # Save to disk
        self._save_rules()
    
    def get_rules_for_action(self, action: str) -> List[BehaviorRule]:
        """
        Get all behavior rules that apply to an action.
        
        Args:
            action: The action/tool being executed
            
        Returns:
            List of applicable behavior rules
        """
        action_lower = action.lower()
        applicable = []
        
        # Check direct matches
        rule_ids = self._trigger_index.get(action_lower, [])
        
        # Check partial matches
        for trigger, ids in self._trigger_index.items():
            if trigger in action_lower or action_lower in trigger:
                rule_ids.extend(ids)
        
        # Get unique rules
        seen = set()
        for rule_id in rule_ids:
            if rule_id not in seen and rule_id in self._rules:
                rule = self._rules[rule_id]
                if rule.enabled:
                    applicable.append(rule)
                    seen.add(rule_id)
        
        return applicable
    
    def get_actions_for(self, trigger_action: str) -> List[BehaviorAction]:
        """
        Get all actions to perform for a trigger.
        
        This is the main method called during tool execution.
        
        Args:
            trigger_action: The tool about to be executed
            
        Returns:
            List of BehaviorActions to execute
        """
        rules = self.get_rules_for_action(trigger_action)
        
        before_actions = []
        after_actions = []
        with_actions = []
        instead_actions = []
        
        for rule in rules:
            rule.use_count += 1
            for action in rule.actions:
                if action.timing == ActionTiming.BEFORE:
                    before_actions.append(action)
                elif action.timing == ActionTiming.AFTER:
                    after_actions.append(action)
                elif action.timing == ActionTiming.WITH:
                    with_actions.append(action)
                elif action.timing == ActionTiming.INSTEAD:
                    instead_actions.append(action)
        
        # If there's an INSTEAD action, that takes precedence
        if instead_actions:
            return instead_actions
        
        # Otherwise return in order: before -> with -> after
        return before_actions + with_actions + after_actions
    
    def get_before_actions(self, trigger_action: str) -> List[BehaviorAction]:
        """Get only BEFORE actions for a trigger."""
        return [
            a for a in self.get_actions_for(trigger_action)
            if a.timing == ActionTiming.BEFORE
        ]
    
    def get_after_actions(self, trigger_action: str) -> List[BehaviorAction]:
        """Get only AFTER actions for a trigger."""
        return [
            a for a in self.get_actions_for(trigger_action)
            if a.timing == ActionTiming.AFTER
        ]
    
    def has_instead_action(self, trigger_action: str) -> bool:
        """Check if there's an INSTEAD action that replaces the trigger."""
        rules = self.get_rules_for_action(trigger_action)
        for rule in rules:
            for action in rule.actions:
                if action.timing == ActionTiming.INSTEAD:
                    return True
        return False
    
    def list_rules(self) -> List[BehaviorRule]:
        """Get all behavior rules."""
        return list(self._rules.values())
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a behavior rule."""
        if rule_id in self._rules:
            rule = self._rules.pop(rule_id)
            
            # Remove from indices
            for trigger, ids in list(self._trigger_index.items()):
                if rule_id in ids:
                    ids.remove(rule_id)
                    if not ids:
                        del self._trigger_index[trigger]
            
            self._save_rules()
            logger.info(f"Removed behavior rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a behavior rule without removing it."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            self._save_rules()
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Re-enable a disabled behavior rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            self._save_rules()
            return True
        return False
    
    def clear_rules(self):
        """Remove all behavior rules."""
        self._rules.clear()
        self._trigger_index.clear()
        self._save_rules()
        logger.info("Cleared all behavior rules")
    
    def _save_rules(self):
        """Save rules to disk."""
        try:
            rules_file = self.storage_dir / "behavior_rules.json"
            data = {
                "version": 1,
                "rules": [rule.to_dict() for rule in self._rules.values()]
            }
            with open(rules_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save behavior rules: {e}")
    
    def _load_rules(self):
        """Load rules from disk."""
        try:
            rules_file = self.storage_dir / "behavior_rules.json"
            if rules_file.exists():
                with open(rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for rule_data in data.get("rules", []):
                    try:
                        rule = BehaviorRule.from_dict(rule_data)
                        self._rules[rule.id] = rule
                        
                        # Build index
                        trigger = rule.trigger_action.lower()
                        if trigger not in self._trigger_index:
                            self._trigger_index[trigger] = []
                        self._trigger_index[trigger].append(rule.id)
                        
                        for keyword in rule.trigger_keywords:
                            keyword = keyword.lower()
                            if keyword not in self._trigger_index:
                                self._trigger_index[keyword] = []
                            if rule.id not in self._trigger_index[keyword]:
                                self._trigger_index[keyword].append(rule.id)
                    except Exception as e:
                        logger.warning(f"Failed to load rule: {e}")
                        
                logger.info(f"Loaded {len(self._rules)} behavior rules")
        except Exception as e:
            logger.warning(f"Could not load behavior rules: {e}")


# Singleton instance
_behavior_manager: Optional[BehaviorManager] = None


def get_behavior_manager() -> BehaviorManager:
    """Get the singleton BehaviorManager instance."""
    global _behavior_manager
    if _behavior_manager is None:
        _behavior_manager = BehaviorManager()
    return _behavior_manager


def check_behavior_statement(statement: str) -> bool:
    """
    Quick check if a statement might be teaching a behavior.
    
    Used by ConversationDetector to identify potential behavior rules.
    
    Args:
        statement: User's statement to check
        
    Returns:
        True if this looks like a behavior teaching statement
    """
    statement_lower = statement.lower()
    
    # Check for common behavior teaching phrases
    indicators = [
        "whenever you",
        "when you",
        "before you",
        "after you",
        "instead of",
        "always",
        "remember to",
        "never forget to",
        "make sure to",
    ]
    
    # Check for action + action pattern
    has_indicator = any(ind in statement_lower for ind in indicators)
    
    # Check for tool/action words
    action_words = ["spawn", "use", "create", "hold", "grab", "teleport", 
                    "move", "walk", "run", "jump", "attack", "cast", "shoot"]
    has_action = any(word in statement_lower for word in action_words)
    
    return has_indicator and has_action


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    manager = BehaviorManager()
    
    # Test learning from statements
    test_statements = [
        "Whenever you teleport, spawn a portal gun first",
        "Before you attack, always cast a shield spell",
        "When you eat, hold the food first",
        "After you dance, do a bow",
        "Instead of walking, always run",
        "Remember to spawn fireworks whenever you win",
    ]
    
    print("Testing behavior learning:")
    print("=" * 50)
    
    for statement in test_statements:
        print(f"\nInput: {statement}")
        rule = manager.learn_from_statement(statement)
        if rule:
            print(f"  Trigger: {rule.trigger_action}")
            for action in rule.actions:
                print(f"  Action: {action.timing.value} -> {action.tool_name}")
                if action.params:
                    print(f"  Params: {action.params}")
        else:
            print("  (No rule detected)")
    
    print("\n" + "=" * 50)
    print(f"\nTotal rules learned: {len(manager.list_rules())}")
    
    # Test getting actions for a trigger
    print("\nActions for 'teleport':")
    actions = manager.get_actions_for("teleport")
    for action in actions:
        print(f"  {action.timing.value}: {action.tool_name} {action.params}")
