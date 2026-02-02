#!/usr/bin/env python3
"""
ForgeAI Desktop Pet / Avatar Example
=====================================

Complete example showing how to use ForgeAI's avatar system including:
- Desktop pet that walks along screen edges
- Lip sync for realistic speaking
- Emotion synchronization
- AI-controlled behaviors
- Mouse/window interaction

The desktop pet is a DesktopMate/Shimeji-style companion that can:
- Walk, sit, sleep, wave, dance
- Follow mouse cursor
- React to screen activity
- Speak with synchronized lip movement
- Express emotions

Dependencies:
    pip install PyQt5  # Required for desktop pet GUI

Run: python examples/avatar_example.py
"""

import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# Core Types and Enums
# =============================================================================

class PetState(Enum):
    """Pet behavior states."""
    IDLE = auto()
    WALKING = auto()
    SITTING = auto()
    SLEEPING = auto()
    FALLING = auto()
    WAVING = auto()
    DANCING = auto()
    TALKING = auto()
    FOLLOWING = auto()


class Mood(Enum):
    """Pet mood/emotion states."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    TIRED = "tired"
    CURIOUS = "curious"
    ANGRY = "angry"


@dataclass
class PetConfig:
    """Desktop pet configuration."""
    name: str = "Forge"
    size: int = 64  # Sprite size in pixels
    walk_speed: int = 2  # Pixels per frame
    gravity: float = 0.5
    bounce: float = 0.3
    idle_timeout: float = 5.0  # Seconds before idle animation
    speech_duration: float = 3.0  # How long speech bubbles last
    follow_distance: int = 100  # Distance to follow cursor


# =============================================================================
# Lip Sync System (Simulated)
# =============================================================================

class LipSync:
    """
    Map text/phonemes to mouth shapes (visemes) for realistic speech.
    
    Visemes are mouth positions that correspond to sounds:
    - 'aa' = mouth open wide (ah, father)
    - 'ee' = smile shape (see, me) 
    - 'oo' = round/pucker (food, you)
    - 'oh' = open circle (go, no)
    - 'mm' = closed lips (mom, hmm)
    """
    
    VISEMES = {
        "silence": "mouth_closed",
        "aa": "mouth_open_wide",
        "ee": "mouth_smile",
        "oo": "mouth_round",
        "oh": "mouth_open",
        "mm": "mouth_closed",
        "ff": "mouth_smile",
        "th": "mouth_open",
    }
    
    def __init__(self):
        self.current_viseme = "silence"
    
    def text_to_visemes(self, text: str) -> List[Tuple[str, float]]:
        """
        Convert text to timed viseme sequence.
        
        Args:
            text: Text being spoken
            
        Returns:
            List of (viseme_name, duration) tuples
        """
        visemes = []
        words = text.split()
        
        for word in words:
            word_duration = len(word) * 0.08  # ~80ms per character
            
            # Simple vowel detection
            word_lower = word.lower()
            if any(v in word_lower for v in ['ee', 'ea', 'i']):
                visemes.append(('ee', word_duration * 0.6))
            elif any(v in word_lower for v in ['oo', 'u']):
                visemes.append(('oo', word_duration * 0.6))
            elif any(v in word_lower for v in ['o']):
                visemes.append(('oh', word_duration * 0.6))
            elif any(v in word_lower for v in ['a', 'ah']):
                visemes.append(('aa', word_duration * 0.6))
            else:
                visemes.append(('mm', word_duration * 0.3))
            
            # Brief silence between words
            visemes.append(('silence', 0.05))
        
        return visemes
    
    def get_mouth_shape(self, viseme: str) -> str:
        """Get mouth shape name for a viseme."""
        return self.VISEMES.get(viseme, "mouth_closed")


# =============================================================================
# Emotion Sync System
# =============================================================================

class EmotionSync:
    """
    Synchronize avatar emotions with AI responses.
    
    Analyzes text sentiment to determine appropriate emotions
    and facial expressions.
    """
    
    # Keywords that indicate different emotions
    EMOTION_KEYWORDS = {
        Mood.HAPPY: ['happy', 'great', 'wonderful', 'love', 'yay', 'awesome', 'good', 'nice'],
        Mood.SAD: ['sad', 'sorry', 'unfortunately', 'bad', 'terrible', 'awful'],
        Mood.EXCITED: ['wow', 'amazing', 'incredible', 'exciting', '!', 'cool'],
        Mood.CURIOUS: ['hmm', 'interesting', 'wonder', '?', 'curious', 'what'],
        Mood.TIRED: ['tired', 'sleepy', 'exhausted', 'yawn', 'zzz'],
        Mood.ANGRY: ['angry', 'frustrated', 'annoyed', 'hate', 'ugh'],
    }
    
    def __init__(self):
        self.current_mood = Mood.NEUTRAL
        self.mood_intensity = 0.5
    
    def analyze_text(self, text: str) -> Tuple[Mood, float]:
        """
        Analyze text to determine emotion.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (mood, intensity 0-1)
        """
        text_lower = text.lower()
        
        # Count keyword matches for each emotion
        scores = {mood: 0 for mood in Mood}
        
        for mood, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[mood] += 1
        
        # Find dominant emotion
        max_score = max(scores.values())
        if max_score > 0:
            dominant = max(scores, key=scores.get)
            intensity = min(1.0, max_score * 0.3)
            return dominant, intensity
        
        return Mood.NEUTRAL, 0.5
    
    def blend_moods(self, current: Mood, target: Mood, 
                    blend_factor: float = 0.3) -> Mood:
        """Smoothly transition between moods."""
        # For simplicity, just return target if blend_factor > 0.5
        return target if blend_factor > 0.5 else current


# =============================================================================
# Desktop Pet (Simulated - No GUI)
# =============================================================================

class DesktopPetSimulator:
    """
    Simulated desktop pet for testing without PyQt5.
    
    In real usage, use forge_ai.avatar.desktop_pet.DesktopPet
    which provides a full graphical desktop companion.
    """
    
    def __init__(self, config: Optional[PetConfig] = None):
        self.config = config or PetConfig()
        self.state = PetState.IDLE
        self.mood = Mood.NEUTRAL
        self.position = (100, 100)
        self.velocity = (0, 0)
        self.is_running = False
        
        self.lip_sync = LipSync()
        self.emotion_sync = EmotionSync()
        
        self._speech_queue: List[str] = []
        self._action_history: List[str] = []
    
    def start(self):
        """Start the desktop pet."""
        self.is_running = True
        self._log(f"{self.config.name} appeared on screen!")
        self.state = PetState.IDLE
    
    def stop(self):
        """Hide the desktop pet."""
        self.is_running = False
        self._log(f"{self.config.name} went away")
    
    def _log(self, action: str):
        """Log an action."""
        self._action_history.append(f"[{time.strftime('%H:%M:%S')}] {action}")
        print(f"  Pet: {action}")
    
    # -------------------------------------------------------------------------
    # Movement
    # -------------------------------------------------------------------------
    
    def walk_to(self, x: int, y: int):
        """Walk to a position on screen."""
        self.state = PetState.WALKING
        old_pos = self.position
        self.position = (x, y)
        self._log(f"Walking from {old_pos} to {self.position}")
    
    def jump(self):
        """Make the pet jump."""
        self.velocity = (self.velocity[0], -10)
        self._log("Jumped!")
    
    def fall(self):
        """Start falling (gravity simulation)."""
        self.state = PetState.FALLING
        self._log("Falling...")
    
    # -------------------------------------------------------------------------
    # Behaviors
    # -------------------------------------------------------------------------
    
    def sit(self):
        """Sit down."""
        self.state = PetState.SITTING
        self._log("Sitting down")
    
    def sleep(self):
        """Go to sleep."""
        self.state = PetState.SLEEPING
        self.mood = Mood.TIRED
        self._log("Zzz... sleeping")
    
    def wake_up(self):
        """Wake up from sleep."""
        self.state = PetState.IDLE
        self.mood = Mood.NEUTRAL
        self._log("Woke up!")
    
    def wave(self):
        """Wave at the user."""
        self.state = PetState.WAVING
        self._log("Waving hello!")
    
    def dance(self):
        """Do a little dance."""
        self.state = PetState.DANCING
        self.mood = Mood.HAPPY
        self._log("Dancing!")
    
    def follow_cursor(self, cursor_x: int, cursor_y: int):
        """Follow the mouse cursor."""
        self.state = PetState.FOLLOWING
        dist_x = cursor_x - self.position[0]
        dist_y = cursor_y - self.position[1]
        
        # Move toward cursor
        new_x = self.position[0] + (1 if dist_x > 0 else -1) * self.config.walk_speed
        new_y = self.position[1] + (1 if dist_y > 0 else -1) * self.config.walk_speed
        
        self.position = (new_x, new_y)
        self._log(f"Following cursor to ({cursor_x}, {cursor_y})")
    
    # -------------------------------------------------------------------------
    # Speech
    # -------------------------------------------------------------------------
    
    def say(self, text: str, with_emotion: bool = True):
        """
        Speak with speech bubble and lip sync.
        
        Args:
            text: Text to speak
            with_emotion: Whether to analyze and show emotion
        """
        self.state = PetState.TALKING
        
        # Analyze emotion from text
        if with_emotion:
            self.mood, intensity = self.emotion_sync.analyze_text(text)
        
        # Generate lip sync data
        visemes = self.lip_sync.text_to_visemes(text)
        
        self._log(f"Says: '{text}' (mood: {self.mood.value})")
        self._log(f"  Lip sync: {len(visemes)} visemes")
        
        self._speech_queue.append(text)
    
    def set_mood(self, mood: Mood):
        """Set the pet's mood."""
        self.mood = mood
        self._log(f"Mood changed to {mood.value}")
    
    # -------------------------------------------------------------------------
    # AI Integration
    # -------------------------------------------------------------------------
    
    def respond_to_ai(self, ai_response: str):
        """
        Respond to AI-generated text with appropriate behavior.
        
        This is the main integration point - the AI generates text,
        and the pet animates accordingly.
        """
        # Analyze emotion
        mood, intensity = self.emotion_sync.analyze_text(ai_response)
        self.mood = mood
        
        # Speak with lip sync
        self.say(ai_response, with_emotion=False)  # Already analyzed
        
        # Add behavioral reactions based on mood
        if mood == Mood.HAPPY:
            self._log("Looking happy!")
        elif mood == Mood.EXCITED:
            self.jump()
        elif mood == Mood.SAD:
            self._log("Looking sad...")
        elif mood == Mood.CURIOUS:
            self._log("Tilting head curiously")
    
    def idle_behavior(self):
        """Random idle behavior when nothing else is happening."""
        if self.state not in [PetState.IDLE, PetState.SITTING]:
            return
        
        behavior = random.choice(['look_around', 'stretch', 'blink', 'yawn', 'scratch'])
        self._log(f"Idle: {behavior}")
    
    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------
    
    def get_state(self) -> Dict:
        """Get current pet state."""
        return {
            'name': self.config.name,
            'state': self.state.name,
            'mood': self.mood.value,
            'position': self.position,
            'is_running': self.is_running
        }
    
    def get_history(self) -> List[str]:
        """Get action history."""
        return self._action_history.copy()


# =============================================================================
# Avatar Controller (Priority-Based Control)
# =============================================================================

class ControlPriority:
    """Control priority levels (higher = takes precedence)."""
    BONE_ANIMATION = 100    # Direct bone control (primary)
    USER_MANUAL = 80        # User dragging/clicking
    AI_TOOL_CALL = 70       # AI explicit commands
    AUTONOMOUS = 50         # Autonomous behaviors (fallback)
    IDLE_ANIMATION = 30     # Background animations
    FALLBACK = 10           # For non-avatar-trained models


class AvatarController:
    """
    Central controller for avatar with priority-based control.
    
    Multiple systems can try to control the avatar, but only
    the highest priority action executes.
    """
    
    def __init__(self, pet: DesktopPetSimulator):
        self.pet = pet
        self.current_priority = 0
        self.control_source = "none"
    
    def request_control(self, priority: int, source: str) -> bool:
        """
        Request control of the avatar.
        
        Args:
            priority: Control priority level
            source: Name of requesting system
            
        Returns:
            True if control granted
        """
        if priority >= self.current_priority:
            self.current_priority = priority
            self.control_source = source
            print(f"  Control granted to '{source}' (priority {priority})")
            return True
        else:
            print(f"  Control denied to '{source}' (priority {priority} < {self.current_priority})")
            return False
    
    def release_control(self, source: str):
        """Release control back to lower priority systems."""
        if self.control_source == source:
            self.current_priority = 0
            self.control_source = "none"
            print(f"  Control released by '{source}'")
    
    def execute_action(self, action: str, priority: int, 
                       source: str, **kwargs) -> bool:
        """
        Execute an action if priority is high enough.
        
        Args:
            action: Action name (walk, say, dance, etc.)
            priority: Control priority
            source: Requesting system name
            **kwargs: Action-specific parameters
        """
        if not self.request_control(priority, source):
            return False
        
        # Execute action
        if action == "walk":
            self.pet.walk_to(kwargs.get('x', 0), kwargs.get('y', 0))
        elif action == "say":
            self.pet.say(kwargs.get('text', ''))
        elif action == "dance":
            self.pet.dance()
        elif action == "wave":
            self.pet.wave()
        elif action == "sleep":
            self.pet.sleep()
        elif action == "sit":
            self.pet.sit()
        elif action == "mood":
            self.pet.set_mood(kwargs.get('mood', Mood.NEUTRAL))
        else:
            print(f"  Unknown action: {action}")
            return False
        
        return True


# =============================================================================
# Example Usage
# =============================================================================

def example_basic_pet():
    """Basic desktop pet creation and control."""
    print("\n" + "="*60)
    print("Example 1: Basic Desktop Pet")
    print("="*60)
    
    # Create pet with custom config
    config = PetConfig(
        name="Sparky",
        size=64,
        walk_speed=3
    )
    
    pet = DesktopPetSimulator(config)
    pet.start()
    
    # Basic actions
    pet.wave()
    pet.walk_to(500, 300)
    pet.sit()
    pet.dance()
    pet.sleep()
    
    # Check state
    print(f"\nPet state: {pet.get_state()}")


def example_lip_sync():
    """Lip sync for speaking animations."""
    print("\n" + "="*60)
    print("Example 2: Lip Sync")
    print("="*60)
    
    lip_sync = LipSync()
    
    text = "Hello there! How are you doing today?"
    visemes = lip_sync.text_to_visemes(text)
    
    print(f"Text: '{text}'")
    print(f"Generated {len(visemes)} visemes:")
    
    total_time = 0
    for viseme, duration in visemes[:10]:  # Show first 10
        mouth = lip_sync.get_mouth_shape(viseme)
        print(f"  {total_time:.2f}s: {viseme:10} -> {mouth}")
        total_time += duration


def example_emotion_sync():
    """Emotion detection from text."""
    print("\n" + "="*60)
    print("Example 3: Emotion Sync")
    print("="*60)
    
    emotion_sync = EmotionSync()
    
    test_texts = [
        "I'm so happy to see you!",
        "That's really sad news...",
        "Wow, that's incredible!",
        "Hmm, I wonder what that means?",
        "I'm feeling tired today.",
        "This is frustrating, ugh!",
    ]
    
    for text in test_texts:
        mood, intensity = emotion_sync.analyze_text(text)
        print(f"  '{text[:30]}...'")
        print(f"    -> {mood.value} (intensity: {intensity:.2f})")


def example_ai_integration():
    """AI integration - pet responds to AI output."""
    print("\n" + "="*60)
    print("Example 4: AI Integration")
    print("="*60)
    
    pet = DesktopPetSimulator()
    pet.start()
    
    # Simulate AI responses
    ai_responses = [
        "Hello! I'm so happy to help you today!",
        "Hmm, that's an interesting question. Let me think...",
        "Wow! That's amazing news, congratulations!",
        "I'm sorry to hear that. Is there anything I can do?",
    ]
    
    for response in ai_responses:
        print(f"\nAI Response: '{response[:40]}...'")
        pet.respond_to_ai(response)


def example_priority_control():
    """Priority-based avatar control."""
    print("\n" + "="*60)
    print("Example 5: Priority-Based Control")
    print("="*60)
    
    pet = DesktopPetSimulator()
    pet.start()
    
    controller = AvatarController(pet)
    
    print("\n1. Idle animation tries to control (low priority):")
    controller.execute_action("sit", ControlPriority.IDLE_ANIMATION, "idle_system")
    
    print("\n2. AI tool call takes over (higher priority):")
    controller.execute_action("dance", ControlPriority.AI_TOOL_CALL, "ai_system")
    
    print("\n3. Idle tries again (blocked by AI):")
    controller.execute_action("sleep", ControlPriority.IDLE_ANIMATION, "idle_system")
    
    print("\n4. User manual control takes over (highest):")
    controller.execute_action("wave", ControlPriority.USER_MANUAL, "user")
    
    print("\n5. AI releases control:")
    controller.release_control("ai_system")
    
    print("\n6. Now idle can control:")
    controller.execute_action("sit", ControlPriority.IDLE_ANIMATION, "idle_system")


def example_real_forge():
    """Using actual ForgeAI avatar system."""
    print("\n" + "="*60)
    print("Example 6: Real ForgeAI Integration")
    print("="*60)
    
    print("For actual ForgeAI desktop pet:")
    print("""
    from forge_ai.avatar.desktop_pet import DesktopPet
    from forge_ai.avatar.lip_sync import LipSync
    from forge_ai.avatar.emotion_sync import EmotionSync
    from forge_ai.avatar.controller import AvatarController, ControlPriority
    
    # Create and start desktop pet
    pet = DesktopPet()
    pet.start()  # Opens PyQt5 window with animated sprite
    
    # Control via AI
    pet.say("Hello there!")  # With lip sync
    pet.set_mood("happy")    # Changes expression
    pet.walk_to(500, 300)    # Animated walk
    pet.dance()              # Dance animation
    
    # The pet will also:
    # - Walk along screen edges like DesktopMate
    # - Sit on window title bars
    # - Follow your mouse cursor
    # - Have idle animations when not controlled
    # - Fall with gravity if window closes under it
    
    pet.stop()  # Hide pet
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("ForgeAI Desktop Pet / Avatar Examples")
    print("="*60)
    
    example_basic_pet()
    example_lip_sync()
    example_emotion_sync()
    example_ai_integration()
    example_priority_control()
    example_real_forge()
    
    print("\n" + "="*60)
    print("Avatar System Summary:")
    print("="*60)
    print("""
Key Components:

1. DesktopPet:
   - Animated sprite on your desktop
   - Walks along edges like DesktopMate/Shimeji
   - Sits on windows, follows cursor
   - Has idle behaviors (sleep, wave, dance)
   - Gravity simulation when falling

2. LipSync:
   - Converts text to mouth shapes (visemes)
   - Realistic speaking animation
   - Maps phonemes to mouth positions

3. EmotionSync:
   - Detects emotion from AI text
   - Changes avatar expression
   - Happy, sad, excited, curious, etc.

4. AvatarController:
   - Priority-based control system
   - Prevents conflicting commands
   - AI, user, and autonomous control layers

5. Control Priority:
   - BONE_ANIMATION (100) - Primary control
   - USER_MANUAL (80) - User clicking/dragging
   - AI_TOOL_CALL (70) - AI explicit commands  
   - AUTONOMOUS (50) - Automatic behaviors
   - IDLE_ANIMATION (30) - Background animations

For real desktop pet:
    pip install PyQt5
    from forge_ai.avatar.desktop_pet import DesktopPet
""")
