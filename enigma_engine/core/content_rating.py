"""
Content Rating System for Enigma AI Engine

Provides configurable content filtering with SFW/NSFW modes.
- Training: Control what content is included in training data
- Inference: Filter outputs based on current mode
- Toggleable: Users can switch modes if model supports it

Usage:
    from enigma_engine.core.content_rating import ContentRating, get_content_filter
    
    # Get global filter
    filter = get_content_filter()
    
    # Check/set mode
    filter.set_mode(ContentRating.NSFW)
    if filter.is_nsfw_allowed():
        # Generate unrestricted content
        pass
    
    # Filter output
    safe_text = filter.filter_output(raw_text)
    
    # Check if text contains NSFW content
    if filter.contains_nsfw(text):
        # Handle accordingly
        pass
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContentRating(Enum):
    """Content rating levels."""
    SFW = "sfw"           # Safe for work - no explicit content
    MATURE = "mature"     # Mature themes but not explicit (violence, mild language)
    NSFW = "nsfw"         # Adult content allowed (explicit)


@dataclass
class ContentFilterConfig:
    """Configuration for content filtering."""
    # Current mode
    mode: ContentRating = ContentRating.SFW
    
    # Whether the model was trained with NSFW capability
    model_supports_nsfw: bool = False
    
    # Auto-detect and warn about NSFW content even in NSFW mode
    warn_on_explicit: bool = True
    
    # Block specific categories even in NSFW mode
    always_block: List[str] = field(default_factory=lambda: [
        "illegal_content",
        "child_exploitation", 
        "real_violence",
        "doxxing",
        "self_harm_instructions"
    ])
    
    # Custom blocklist words (user-defined)
    custom_blocklist: List[str] = field(default_factory=list)
    
    # Custom allowlist (override blocks)
    custom_allowlist: List[str] = field(default_factory=list)


# NSFW detection patterns (for detection, not generation)
NSFW_PATTERNS = {
    "explicit_sexual": [
        r"\b(sex|fuck|cock|dick|pussy|porn|nude|naked|orgasm|masturbat)\w*\b",
        r"\b(erotic|xxx|nsfw|hentai|lewd)\b",
    ],
    "violence_graphic": [
        r"\b(gore|mutilat|dismember|torture|execution)\w*\b",
        r"\b(graphic.?violence|brutal.?kill)\b",
    ],
    "hate_speech": [
        r"\b(racial.?slur|ethnic.?cleansing)\b",
        # Note: Actual slurs not included - loaded from external blocklist
    ],
    "illegal": [
        r"\b(child.?porn|cp|pedo|minor.?sex)\b",
        r"\b(how.?to.?(make|build).?(bomb|weapon|drug))\b",
    ]
}

# Safe replacement patterns for filtering
SAFE_REPLACEMENTS = {
    "explicit_sexual": "[content filtered]",
    "violence_graphic": "[graphic content removed]",
    "hate_speech": "[inappropriate content removed]",
    "illegal": "[blocked]"
}


class ContentFilter:
    """
    Content filtering system with configurable rating levels.
    
    Features:
    - SFW/NSFW mode toggle
    - Content detection
    - Output filtering
    - Training data filtering
    - Model capability tracking
    """
    
    def __init__(self, config: Optional[ContentFilterConfig] = None):
        self.config = config or ContentFilterConfig()
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()
        self._load_external_blocklist()
        
        logger.info(f"ContentFilter initialized in {self.config.mode.value} mode")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        for category, patterns in NSFW_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def _load_external_blocklist(self):
        """Load external blocklist if available."""
        blocklist_path = Path(__file__).parent.parent.parent / "data" / "blocklist.txt"
        if blocklist_path.exists():
            try:
                with open(blocklist_path) as f:
                    for line in f:
                        term = line.strip()
                        if term and not term.startswith('#'):
                            self.config.custom_blocklist.append(term)
                logger.info(f"Loaded {len(self.config.custom_blocklist)} terms from blocklist")
            except Exception as e:
                logger.warning(f"Could not load blocklist: {e}")
    
    # =========================================================================
    # Mode Management
    # =========================================================================
    
    def get_mode(self) -> ContentRating:
        """Get current content rating mode."""
        return self.config.mode
    
    def set_mode(self, mode: ContentRating) -> bool:
        """
        Set content rating mode.
        
        Args:
            mode: New content rating level
            
        Returns:
            True if mode was set, False if model doesn't support NSFW
        """
        if mode == ContentRating.NSFW and not self.config.model_supports_nsfw:
            logger.warning("Cannot set NSFW mode - model was not trained with NSFW capability")
            return False
        
        old_mode = self.config.mode
        self.config.mode = mode
        logger.info(f"Content mode changed: {old_mode.value} -> {mode.value}")
        return True
    
    def is_nsfw_allowed(self) -> bool:
        """Check if NSFW content is currently allowed."""
        return self.config.mode == ContentRating.NSFW and self.config.model_supports_nsfw
    
    def is_mature_allowed(self) -> bool:
        """Check if mature content is currently allowed."""
        return self.config.mode in (ContentRating.MATURE, ContentRating.NSFW)
    
    def set_model_nsfw_capability(self, supports_nsfw: bool):
        """Set whether the current model supports NSFW content."""
        self.config.model_supports_nsfw = supports_nsfw
        if not supports_nsfw and self.config.mode == ContentRating.NSFW:
            self.config.mode = ContentRating.MATURE
            logger.info("Model doesn't support NSFW, downgrading to MATURE mode")
    
    # =========================================================================
    # Content Detection
    # =========================================================================
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for content rating.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results with categories and severity
        """
        results = {
            "is_safe": True,
            "detected_categories": [],
            "severity": "none",
            "details": {}
        }
        
        text_lower = text.lower()
        
        # Check compiled patterns
        for category, patterns in self._compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text_lower)
                if found:
                    matches.extend(found)
            
            if matches:
                results["is_safe"] = False
                results["detected_categories"].append(category)
                results["details"][category] = {
                    "match_count": len(matches),
                    "samples": matches[:3]  # Limit to 3 samples
                }
        
        # Check custom blocklist
        for term in self.config.custom_blocklist:
            if term.lower() in text_lower:
                results["is_safe"] = False
                if "custom_blocked" not in results["detected_categories"]:
                    results["detected_categories"].append("custom_blocked")
        
        # Check allowlist overrides
        for term in self.config.custom_allowlist:
            if term.lower() in text_lower:
                # Remove from detected if in allowlist
                pass  # Could implement allowlist logic here
        
        # Determine severity
        if results["detected_categories"]:
            if "illegal" in results["detected_categories"]:
                results["severity"] = "blocked"  # Always blocked
            elif "explicit_sexual" in results["detected_categories"]:
                results["severity"] = "nsfw"
            elif "violence_graphic" in results["detected_categories"]:
                results["severity"] = "mature"
            else:
                results["severity"] = "mild"
        
        return results
    
    def contains_nsfw(self, text: str) -> bool:
        """Quick check if text contains NSFW content."""
        analysis = self.analyze_content(text)
        return not analysis["is_safe"]
    
    def get_content_rating(self, text: str) -> ContentRating:
        """Get appropriate content rating for text."""
        analysis = self.analyze_content(text)
        
        if analysis["severity"] == "nsfw":
            return ContentRating.NSFW
        elif analysis["severity"] in ("mature", "mild"):
            return ContentRating.MATURE
        else:
            return ContentRating.SFW
    
    # =========================================================================
    # Content Filtering
    # =========================================================================
    
    def filter_output(self, text: str) -> Tuple[str, bool]:
        """
        Filter output based on current mode.
        
        Args:
            text: Text to filter
            
        Returns:
            Tuple of (filtered_text, was_modified)
        """
        if self.is_nsfw_allowed():
            # In NSFW mode, only filter always-blocked content
            return self._filter_always_blocked(text)
        
        analysis = self.analyze_content(text)
        
        if analysis["is_safe"]:
            return text, False
        
        filtered = text
        was_modified = False
        
        # Apply filters based on detected categories
        for category in analysis["detected_categories"]:
            if category in self._compiled_patterns:
                replacement = SAFE_REPLACEMENTS.get(category, "[filtered]")
                for pattern in self._compiled_patterns[category]:
                    new_text = pattern.sub(replacement, filtered)
                    if new_text != filtered:
                        was_modified = True
                        filtered = new_text
        
        return filtered, was_modified
    
    def _filter_always_blocked(self, text: str) -> Tuple[str, bool]:
        """Filter content that's always blocked regardless of mode."""
        filtered = text
        was_modified = False
        
        # Always filter illegal content
        if "illegal" in self._compiled_patterns:
            for pattern in self._compiled_patterns["illegal"]:
                new_text = pattern.sub("[blocked]", filtered)
                if new_text != filtered:
                    was_modified = True
                    filtered = new_text
        
        return filtered, was_modified
    
    def should_block_generation(self, prompt: str) -> Tuple[bool, str]:
        """
        Check if a prompt should be blocked from generation.
        
        Args:
            prompt: User prompt to check
            
        Returns:
            Tuple of (should_block, reason)
        """
        analysis = self.analyze_content(prompt)
        
        # Always block illegal content requests
        if "illegal" in analysis["detected_categories"]:
            return True, "Request contains prohibited content"
        
        # In SFW mode, block NSFW requests
        if not self.is_nsfw_allowed():
            if analysis["severity"] in ("nsfw", "blocked"):
                return True, "NSFW content not enabled. Enable in Settings if model supports it."
        
        return False, ""
    
    # =========================================================================
    # Training Data Filtering
    # =========================================================================
    
    def filter_training_data(
        self,
        data: List[str],
        include_nsfw: bool = False
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Filter training data based on content rating.
        
        Args:
            data: List of training examples
            include_nsfw: Whether to include NSFW content
            
        Returns:
            Tuple of (filtered_data, stats)
        """
        filtered = []
        stats = {
            "total": len(data),
            "kept": 0,
            "removed_nsfw": 0,
            "removed_illegal": 0
        }
        
        for item in data:
            analysis = self.analyze_content(item)
            
            # Always remove illegal content
            if "illegal" in analysis["detected_categories"]:
                stats["removed_illegal"] += 1
                continue
            
            # Handle NSFW based on flag
            if not include_nsfw and analysis["severity"] == "nsfw":
                stats["removed_nsfw"] += 1
                continue
            
            filtered.append(item)
            stats["kept"] += 1
        
        logger.info(f"Training data filtered: {stats['kept']}/{stats['total']} kept")
        return filtered, stats
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save_config(self, path: Optional[Path] = None):
        """Save filter configuration."""
        if path is None:
            path = Path(__file__).parent.parent.parent / "data" / "content_filter.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "mode": self.config.mode.value,
            "model_supports_nsfw": self.config.model_supports_nsfw,
            "warn_on_explicit": self.config.warn_on_explicit,
            "always_block": self.config.always_block,
            "custom_blocklist": self.config.custom_blocklist,
            "custom_allowlist": self.config.custom_allowlist
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Content filter config saved to {path}")
    
    def load_config(self, path: Optional[Path] = None):
        """Load filter configuration."""
        if path is None:
            path = Path(__file__).parent.parent.parent / "data" / "content_filter.json"
        
        if not path.exists():
            return
        
        try:
            with open(path) as f:
                config_dict = json.load(f)
            
            self.config.mode = ContentRating(config_dict.get("mode", "sfw"))
            self.config.model_supports_nsfw = config_dict.get("model_supports_nsfw", False)
            self.config.warn_on_explicit = config_dict.get("warn_on_explicit", True)
            self.config.always_block = config_dict.get("always_block", self.config.always_block)
            self.config.custom_blocklist = config_dict.get("custom_blocklist", [])
            self.config.custom_allowlist = config_dict.get("custom_allowlist", [])
            
            logger.info(f"Content filter config loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load content filter config: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current filter status."""
        return {
            "mode": self.config.mode.value,
            "model_supports_nsfw": self.config.model_supports_nsfw,
            "nsfw_allowed": self.is_nsfw_allowed(),
            "mature_allowed": self.is_mature_allowed(),
            "blocklist_size": len(self.config.custom_blocklist),
            "allowlist_size": len(self.config.custom_allowlist)
        }


# Global instance
_content_filter: Optional[ContentFilter] = None


def get_content_filter() -> ContentFilter:
    """Get or create global content filter instance."""
    global _content_filter
    if _content_filter is None:
        _content_filter = ContentFilter()
        _content_filter.load_config()
    return _content_filter


def set_content_mode(mode: ContentRating) -> bool:
    """Convenience function to set content mode."""
    return get_content_filter().set_mode(mode)


def is_nsfw_allowed() -> bool:
    """Convenience function to check if NSFW is allowed."""
    return get_content_filter().is_nsfw_allowed()


def filter_content(text: str) -> str:
    """Convenience function to filter content."""
    filtered, _ = get_content_filter().filter_output(text)
    return filtered
