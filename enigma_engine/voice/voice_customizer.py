"""
Voice Customizer - Interactive User Tools

Interactive tools for users to customize voice profiles.

Features:
- Interactive voice tuning with previews
- Import/export voice profiles
- Voice comparison tool
- Voice preset exploration

Usage:
    from enigma_engine.voice.voice_customizer import VoiceCustomizer
    
    customizer = VoiceCustomizer()
    
    # Interactive tuning (CLI or GUI)
    profile = customizer.interactive_tuning()
    
    # Import/export
    customizer.export_profile(profile, "my_voice.json")
    profile = customizer.import_profile("my_voice.json")
    
    # Compare voices
    customizer.compare_voices([profile1, profile2], "Test text")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .voice_effects import VoiceEffects
from .voice_profile import PRESET_PROFILES, VoiceEngine, VoiceProfile

logger = logging.getLogger(__name__)


class VoiceCustomizer:
    """
    Interactive tools for voice customization.
    
    Provides helpers for users to create and tune voice profiles.
    """
    
    def __init__(self):
        """Initialize voice customizer."""
        self.engine = VoiceEngine()
        self.effects = VoiceEffects()
    
    def interactive_tuning(
        self,
        base_profile: Optional[VoiceProfile] = None,
        preview_text: str = "Hello, this is a test of my voice."
    ) -> VoiceProfile:
        """
        Step-by-step voice tuning with previews.
        
        This is a CLI-based interactive tuner. For GUI, integrate with PyQt5.
        
        Args:
            base_profile: Starting profile (uses default if None)
            preview_text: Text to preview voice changes
            
        Returns:
            Customized VoiceProfile
        """
        if base_profile is None:
            base_profile = VoiceProfile()
        
        print("\n=== Voice Customizer ===")
        print("Adjust voice parameters. Type 'preview' to hear it, 'done' when finished.\n")
        
        # Current settings
        current = VoiceProfile(
            name=base_profile.name,
            pitch=base_profile.pitch,
            speed=base_profile.speed,
            volume=base_profile.volume,
            voice=base_profile.voice,
            effects=base_profile.effects.copy(),
            description=base_profile.description
        )
        
        while True:
            self._print_current_settings(current)
            
            command = input("\nCommand (pitch/speed/volume/voice/effects/preview/done): ").strip().lower()
            
            if command == "done":
                break
            elif command == "preview":
                self._preview_voice(current, preview_text)
            elif command == "pitch":
                current.pitch = self._adjust_parameter(
                    "Pitch", current.pitch, 0.5, 1.5, 0.1
                )
            elif command == "speed":
                current.speed = self._adjust_parameter(
                    "Speed", current.speed, 0.5, 1.5, 0.1
                )
            elif command == "volume":
                current.volume = self._adjust_parameter(
                    "Volume", current.volume, 0.3, 1.0, 0.1
                )
            elif command == "voice":
                current.voice = self._select_voice()
            elif command == "effects":
                current.effects = self._select_effects()
            else:
                print("Unknown command. Try: pitch, speed, volume, voice, effects, preview, done")
        
        # Finalize
        name = input("\nEnter name for this voice profile: ").strip()
        if name:
            current.name = name
        
        description = input("Enter description (optional): ").strip()
        if description:
            current.description = description
        
        print(f"\n✓ Voice profile '{current.name}' created!")
        logger.debug(f"Voice profile created: {current.name} (pitch={current.pitch:.2f}, speed={current.speed:.2f})")
        return current
    
    def _print_current_settings(self, profile: VoiceProfile):
        """Print current voice settings."""
        print("\n--- Current Settings ---")
        print(f"Name:    {profile.name}")
        print(f"Pitch:   {profile.pitch:.2f} (0.5-1.5)")
        print(f"Speed:   {profile.speed:.2f} (0.5-1.5)")
        print(f"Volume:  {profile.volume:.2f} (0.3-1.0)")
        print(f"Voice:   {profile.voice}")
        print(f"Effects: {', '.join(profile.effects) if profile.effects else 'none'}")
    
    def _adjust_parameter(
        self,
        name: str,
        current: float,
        min_val: float,
        max_val: float,
        step: float
    ) -> float:
        """Interactive parameter adjustment."""
        print(f"\n{name}: {current:.2f} (range: {min_val}-{max_val})")
        print("Commands: +/- to adjust, or enter exact value")
        
        while True:
            cmd = input(f"{name} > ").strip()
            
            if cmd == "+":
                new_val = min(max_val, current + step)
                print(f"  → {new_val:.2f}")
                return new_val
            elif cmd == "-":
                new_val = max(min_val, current - step)
                print(f"  → {new_val:.2f}")
                return new_val
            elif cmd == "":
                return current
            else:
                try:
                    new_val = float(cmd)
                    if min_val <= new_val <= max_val:
                        print(f"  → {new_val:.2f}")
                        return new_val
                    else:
                        print(f"  Value must be between {min_val} and {max_val}")
                except ValueError:
                    print("  Invalid input. Use +, -, or a number.")
    
    def _select_voice(self) -> str:
        """Interactive voice selection."""
        # Get available system voices
        voices = self.engine.get_available_voices()
        
        print("\nAvailable voices:")
        print("  1. default")
        print("  2. male")
        print("  3. female")
        
        if voices:
            print("\nSystem voices:")
            for i, voice in enumerate(voices[:5], start=4):
                print(f"  {i}. {voice['name']}")
        
        choice = input("\nSelect voice (number or name): ").strip()
        
        if choice == "1":
            return "default"
        elif choice == "2":
            return "male"
        elif choice == "3":
            return "female"
        elif choice.isdigit():
            idx = int(choice) - 4
            if 0 <= idx < len(voices):
                return voices[idx]["id"]
        
        return choice  # Assume it's a voice name/ID
    
    def _select_effects(self) -> List[str]:
        """Interactive effects selection."""
        available = self.effects.list_available_effects()
        
        print("\nAvailable effects:")
        for i, effect in enumerate(available, start=1):
            desc = self.effects.get_effect_description(effect)
            print(f"  {i}. {effect} - {desc}")
        
        print("\nEnter effect numbers separated by spaces (or 'none'):")
        choice = input("Effects > ").strip()
        
        if choice.lower() == "none":
            return []
        
        selected = []
        for num_str in choice.split():
            try:
                idx = int(num_str) - 1
                if 0 <= idx < len(available):
                    selected.append(available[idx])
            except ValueError:
                pass  # Intentionally silent
        
        return selected
    
    def _preview_voice(self, profile: VoiceProfile, text: str):
        """Preview voice with current settings."""
        print(f"\nPreviewing: '{text}'")
        self.engine.set_profile(profile)
        self.engine.speak(text)
    
    def import_profile(self, path: str) -> VoiceProfile:
        """
        Import voice profile from file.
        
        Args:
            path: Path to JSON profile file
            
        Returns:
            Loaded VoiceProfile
        """
        filepath = Path(path)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")
        
        with open(filepath) as f:
            data = json.load(f)
        
        profile = VoiceProfile(**data)
        print(f"✓ Imported voice profile '{profile.name}' from {path}")
        logger.debug(f"Imported voice profile: {profile.name} from {path}")
        
        return profile
    
    def export_profile(self, profile: VoiceProfile, path: str) -> Path:
        """
        Export voice profile to file.
        
        Args:
            profile: VoiceProfile to export
            path: Destination file path
            
        Returns:
            Path to exported file
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(profile)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported voice profile to {filepath}")
        logger.debug(f"Exported voice profile '{profile.name}' to {filepath}")
        return filepath
    
    def compare_voices(
        self,
        profiles: List[VoiceProfile],
        text: str = "Hello, this is a voice comparison test."
    ):
        """
        Preview multiple voices side-by-side.
        
        Args:
            profiles: List of VoiceProfiles to compare
            text: Text to speak with each voice
        """
        print("\n=== Voice Comparison ===")
        print(f"Text: '{text}'\n")
        
        for i, profile in enumerate(profiles, start=1):
            print(f"{i}. {profile.name}")
            print(f"   Pitch: {profile.pitch:.2f}, Speed: {profile.speed:.2f}, Volume: {profile.volume:.2f}")
            print(f"   Effects: {', '.join(profile.effects) if profile.effects else 'none'}")
            
            input(f"   Press Enter to hear voice {i}...")
            self.engine.set_profile(profile)
            self.engine.speak(text)
            print()
    
    def explore_presets(self, preview_text: str = "Hello, I am a preset voice."):
        """
        Explore and preview all preset voices.
        
        Args:
            preview_text: Text to preview each preset
        """
        print("\n=== Voice Presets ===")
        presets = list(PRESET_PROFILES.items())
        
        for i, (name, profile) in enumerate(presets, start=1):
            print(f"\n{i}. {profile.name}")
            print(f"   {profile.description}")
            print(f"   Pitch: {profile.pitch:.2f}, Speed: {profile.speed:.2f}")
            
            choice = input("   Preview? (y/n): ").strip().lower()
            if choice == 'y':
                self.engine.set_profile(profile)
                self.engine.speak(preview_text)
    
    def create_from_preset(
        self,
        preset_name: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> VoiceProfile:
        """
        Create custom voice starting from a preset.
        
        Args:
            preset_name: Name of preset to start from
            customizations: Dict of parameters to override
            
        Returns:
            Customized VoiceProfile
        """
        if preset_name.lower() not in PRESET_PROFILES:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        base = PRESET_PROFILES[preset_name.lower()]
        
        # Create copy
        custom = VoiceProfile(
            name=base.name,
            pitch=base.pitch,
            speed=base.speed,
            volume=base.volume,
            voice=base.voice,
            effects=base.effects.copy(),
            language=base.language,
            description=base.description
        )
        
        # Apply customizations
        if customizations:
            for key, value in customizations.items():
                if hasattr(custom, key):
                    setattr(custom, key, value)
        
        return custom
    
    def batch_create_variations(
        self,
        base_profile: VoiceProfile,
        num_variations: int = 5
    ) -> List[VoiceProfile]:
        """
        Create multiple variations of a voice profile.
        
        Useful for finding the perfect voice by exploring the parameter space.
        
        Args:
            base_profile: Base profile to vary
            num_variations: Number of variations to create
            
        Returns:
            List of VoiceProfile variations
        """
        variations = []
        
        # Systematic exploration instead of random - explore parameter space evenly
        # Create variations along pitch and speed axes
        pitch_offsets = [-0.15, -0.075, 0, 0.075, 0.15]
        speed_offsets = [-0.15, -0.075, 0, 0.075, 0.15]
        
        i = 0
        for p_off in pitch_offsets:
            for s_off in speed_offsets:
                if i >= num_variations:
                    break
                variation = VoiceProfile(
                    name=f"{base_profile.name}_var{i+1}",
                    pitch=max(0.5, min(1.5, base_profile.pitch + p_off)),
                    speed=max(0.5, min(1.5, base_profile.speed + s_off)),
                    volume=base_profile.volume,  # Keep volume consistent
                    voice=base_profile.voice,
                    effects=base_profile.effects.copy(),
                    language=base_profile.language,
                    description=f"Variation {i+1}: pitch {p_off:+.2f}, speed {s_off:+.2f}"
                )
                variations.append(variation)
                i += 1
            if i >= num_variations:
                break
        
        return variations


# Convenience functions
def interactive_tuning(base_profile: Optional[VoiceProfile] = None) -> VoiceProfile:
    """
    Start interactive voice tuning.
    
    Args:
        base_profile: Starting profile
        
    Returns:
        Customized VoiceProfile
    """
    customizer = VoiceCustomizer()
    return customizer.interactive_tuning(base_profile)


def import_voice_profile(path: str) -> VoiceProfile:
    """Import voice profile from file."""
    customizer = VoiceCustomizer()
    return customizer.import_profile(path)


def export_voice_profile(profile: VoiceProfile, path: str) -> Path:
    """Export voice profile to file."""
    customizer = VoiceCustomizer()
    return customizer.export_profile(profile, path)


def compare_voices(profiles: List[VoiceProfile], text: str = "Test"):
    """Compare multiple voice profiles."""
    customizer = VoiceCustomizer()
    customizer.compare_voices(profiles, text)
