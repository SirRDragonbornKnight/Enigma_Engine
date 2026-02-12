"""
Multiplayer Awareness for Enigma AI Engine

Detect and track other players in multiplayer games.

Features:
- Player detection and tracking
- Team awareness
- Voice chat integration
- Coordinate sharing
- Friend/foe identification
- Party management

Usage:
    from enigma_engine.tools.multiplayer_awareness import MultiplayerAwareness
    
    mp = MultiplayerAwareness()
    
    # Detect players
    players = mp.detect_players(screenshot)
    
    # Track team
    team_info = mp.get_team_status()
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PlayerRelation(Enum):
    """Relationship to detected player."""
    SELF = "self"
    TEAMMATE = "teammate"
    ENEMY = "enemy"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"
    FRIEND = "friend"        # On friends list
    PARTY_MEMBER = "party"   # In same party


class PlayerStatus(Enum):
    """Player status."""
    ALIVE = "alive"
    DEAD = "dead"
    AFK = "afk"
    DISCONNECTED = "disconnected"
    SPECTATING = "spectating"
    UNKNOWN = "unknown"


class CommunicationType(Enum):
    """Communication types."""
    TEXT_CHAT = "text"
    VOICE_CHAT = "voice"
    PING = "ping"
    EMOTE = "emote"
    QUICK_CHAT = "quick"


@dataclass
class Position:
    """3D position."""
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position."""
        return (
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        ) ** 0.5


@dataclass
class Player:
    """Detected player information."""
    id: str
    name: str
    relation: PlayerRelation = PlayerRelation.UNKNOWN
    status: PlayerStatus = PlayerStatus.UNKNOWN
    position: Optional[Position] = None
    health: Optional[float] = None
    level: Optional[int] = None
    team: Optional[str] = None
    class_type: Optional[str] = None  # Character class/role
    score: int = 0
    kills: int = 0
    deaths: int = 0
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_hostile(self) -> bool:
        """Check if player is hostile."""
        return self.relation == PlayerRelation.ENEMY
        
    def is_friendly(self) -> bool:
        """Check if player is friendly."""
        return self.relation in (
            PlayerRelation.SELF,
            PlayerRelation.TEAMMATE,
            PlayerRelation.FRIEND,
            PlayerRelation.PARTY_MEMBER
        )


@dataclass
class TeamInfo:
    """Team information."""
    id: str
    name: str
    color: Optional[str] = None
    score: int = 0
    players: List[Player] = field(default_factory=list)
    objectives_held: int = 0


@dataclass 
class GameMessage:
    """In-game message."""
    sender: Optional[Player]
    content: str
    message_type: CommunicationType
    timestamp: float = field(default_factory=time.time)
    is_system: bool = False
    channel: str = "all"  # all, team, party, whisper


@dataclass
class PingMarker:
    """Map ping/marker."""
    position: Position
    ping_type: str  # danger, help, enemy, objective, etc.
    creator: Optional[Player] = None
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    message: Optional[str] = None


class PlayerDetector:
    """Detect players from game screen."""
    
    def __init__(self):
        self.name_patterns = [
            r'\[(\w+)\]\s*(\w+)',         # [Tag] Name
            r'(\w+)\s*\((\w+)\)',          # Name (Class)
            r'<(\w+)>\s*(\w+)',            # <Guild> Name
            r'(\w+)\s*Lv\.?\s*(\d+)',      # Name Lv.50
        ]
        
    def detect_from_screen(
        self,
        image: Any,
        regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    ) -> List[Player]:
        """Detect players from screenshot."""
        players = []
        
        try:
            # Try OCR detection
            from PIL import Image
            
            if isinstance(image, str):
                image = Image.open(image)
            elif hasattr(image, 'copy'):
                image = image.copy()
                
            # Default regions for common UI elements
            if regions is None:
                regions = {
                    'scoreboard': (0, 0, image.width, image.height // 3),
                    'player_list': (image.width - 300, 0, image.width, image.height),
                    'minimap': (image.width - 200, image.height - 200, image.width, image.height)
                }
                
            # Process each region
            for region_name, bbox in regions.items():
                region_players = self._process_region(image.crop(bbox), region_name)
                players.extend(region_players)
                
        except ImportError:
            logger.warning("PIL not available for player detection")
        except Exception as e:
            logger.error(f"Player detection failed: {e}")
            
        return players
        
    def _process_region(self, image: Any, region_name: str) -> List[Player]:
        """Process a screen region for players."""
        players = []
        
        try:
            import pytesseract
            
            text = pytesseract.image_to_string(image)
            
            # Extract player names using patterns
            for pattern in self.name_patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1) if len(match.groups()) >= 1 else "Unknown"
                    player = Player(
                        id=hashlib.md5(name.encode()).hexdigest()[:8],
                        name=name
                    )
                    
                    # Extract level if present
                    if len(match.groups()) >= 2:
                        try:
                            player.level = int(match.group(2))
                        except ValueError:
                            pass  # Intentionally silent
                            
                    players.append(player)
                    
        except ImportError:
            logger.debug("pytesseract not available")
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            
        return players
        
    def detect_from_minimap(
        self,
        minimap_image: Any,
        icon_templates: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Position, PlayerRelation]]:
        """Detect players from minimap icons."""
        detections = []
        
        try:
            import cv2
            import numpy as np
            
            if isinstance(minimap_image, str):
                minimap = cv2.imread(minimap_image)
            else:
                minimap = np.array(minimap_image)
                
            # Default color-based detection
            # Red = enemy, Green/Blue = friendly
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Red detection (enemy)
            red_lower = np.array([0, 100, 100])
            red_upper = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, red_lower, red_upper)
            
            red_contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in red_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections.append((
                        Position(cx, cy),
                        PlayerRelation.ENEMY
                    ))
                    
            # Green detection (friendly)
            green_lower = np.array([40, 100, 100])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            green_contours, _ = cv2.findContours(
                green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in green_contours:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections.append((
                        Position(cx, cy),
                        PlayerRelation.TEAMMATE
                    ))
                    
        except ImportError:
            logger.debug("OpenCV not available for minimap detection")
        except Exception as e:
            logger.debug(f"Minimap detection failed: {e}")
            
        return detections


class ChatMonitor:
    """Monitor in-game chat."""
    
    def __init__(self):
        self.message_history: List[GameMessage] = []
        self.max_history = 100
        
        # Patterns for chat parsing
        self.chat_patterns = {
            'team': r'\[(?:Team|Equipe|팀)\]\s*(\w+):\s*(.+)',
            'all': r'\[(?:All|Global|全部)\]\s*(\w+):\s*(.+)',
            'party': r'\[(?:Party|Groupe|파티)\]\s*(\w+):\s*(.+)',
            'whisper': r'(?:From|De|에게서)\s*\[?(\w+)\]?:\s*(.+)',
            'system': r'\[(?:System|Systeme|시스템)\]\s*(.+)',
        }
        
    def parse_chat_text(self, text: str) -> List[GameMessage]:
        """Parse chat text from OCR or log."""
        messages = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            for channel, pattern in self.chat_patterns.items():
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if channel == 'system':
                        msg = GameMessage(
                            sender=None,
                            content=match.group(1),
                            message_type=CommunicationType.TEXT_CHAT,
                            is_system=True,
                            channel=channel
                        )
                    else:
                        sender_name = match.group(1)
                        msg = GameMessage(
                            sender=Player(
                                id=hashlib.md5(sender_name.encode()).hexdigest()[:8],
                                name=sender_name
                            ),
                            content=match.group(2),
                            message_type=CommunicationType.TEXT_CHAT,
                            channel=channel
                        )
                    messages.append(msg)
                    break
                    
        self.message_history.extend(messages)
        
        # Trim history
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]
            
        return messages
        
    def get_recent_messages(
        self,
        count: int = 10,
        channel: Optional[str] = None
    ) -> List[GameMessage]:
        """Get recent chat messages."""
        messages = self.message_history
        
        if channel:
            messages = [m for m in messages if m.channel == channel]
            
        return messages[-count:]
        
    def find_mentions(self, player_name: str) -> List[GameMessage]:
        """Find messages mentioning a player."""
        return [
            m for m in self.message_history
            if player_name.lower() in m.content.lower()
        ]


class MultiplayerAwareness:
    """Main multiplayer awareness system."""
    
    def __init__(self):
        self.detector = PlayerDetector()
        self.chat = ChatMonitor()
        
        self.known_players: Dict[str, Player] = {}
        self.teams: Dict[str, TeamInfo] = {}
        self.pings: List[PingMarker] = []
        self.friends_list: Set[str] = set()
        self.party_members: Set[str] = set()
        
        # Self player info
        self.self_player: Optional[Player] = None
        
        # Callbacks
        self.on_player_detected: List[Callable[[Player], None]] = []
        self.on_player_killed: List[Callable[[Player, Optional[Player]], None]] = []
        self.on_message: List[Callable[[GameMessage], None]] = []
        self.on_ping: List[Callable[[PingMarker], None]] = []
        
    def set_self(self, name: str, **kwargs) -> Player:
        """Set self player info."""
        self.self_player = Player(
            id="self",
            name=name,
            relation=PlayerRelation.SELF,
            status=PlayerStatus.ALIVE,
            **kwargs
        )
        self.known_players["self"] = self.self_player
        return self.self_player
        
    def add_friend(self, name: str):
        """Add player to friends list."""
        self.friends_list.add(name.lower())
        
    def add_party_member(self, name: str):
        """Add player to party."""
        self.party_members.add(name.lower())
        
    def detect_players(
        self,
        screenshot: Any,
        regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    ) -> List[Player]:
        """Detect players from screenshot."""
        detected = self.detector.detect_from_screen(screenshot, regions)
        
        for player in detected:
            # Classify relationship
            if player.name.lower() in self.friends_list:
                player.relation = PlayerRelation.FRIEND
            elif player.name.lower() in self.party_members:
                player.relation = PlayerRelation.PARTY_MEMBER
            elif self.self_player and player.team == self.self_player.team:
                player.relation = PlayerRelation.TEAMMATE
                
            # Update known players
            existing = self.known_players.get(player.id)
            if existing:
                # Update existing
                existing.position = player.position or existing.position
                existing.health = player.health or existing.health
                existing.last_seen = time.time()
            else:
                # New player
                self.known_players[player.id] = player
                for callback in self.on_player_detected:
                    callback(player)
                    
        return detected
        
    def update_from_minimap(self, minimap_image: Any) -> List[Player]:
        """Update player positions from minimap."""
        detections = self.detector.detect_from_minimap(minimap_image)
        
        updated = []
        for position, relation in detections:
            # Create or update player
            player_id = f"minimap_{position.x}_{position.y}"
            
            if player_id not in self.known_players:
                player = Player(
                    id=player_id,
                    name=f"Player@{position.x},{position.y}",
                    relation=relation,
                    position=position
                )
                self.known_players[player_id] = player
            else:
                player = self.known_players[player_id]
                player.position = position
                player.last_seen = time.time()
                
            updated.append(player)
            
        return updated
        
    def process_chat(self, chat_text: str) -> List[GameMessage]:
        """Process chat text."""
        messages = self.chat.parse_chat_text(chat_text)
        
        for msg in messages:
            for callback in self.on_message:
                callback(msg)
                
        return messages
        
    def add_ping(
        self,
        position: Position,
        ping_type: str,
        message: Optional[str] = None,
        duration: float = 10.0
    ) -> PingMarker:
        """Add a map ping."""
        ping = PingMarker(
            position=position,
            ping_type=ping_type,
            creator=self.self_player,
            message=message,
            expires_at=time.time() + duration if duration > 0 else None
        )
        
        self.pings.append(ping)
        
        for callback in self.on_ping:
            callback(ping)
            
        return ping
        
    def get_active_pings(self) -> List[PingMarker]:
        """Get active (non-expired) pings."""
        now = time.time()
        active = [
            p for p in self.pings
            if p.expires_at is None or p.expires_at > now
        ]
        self.pings = active
        return active
        
    def get_nearby_players(
        self,
        max_distance: float = 100.0,
        relation: Optional[PlayerRelation] = None
    ) -> List[Player]:
        """Get players near self."""
        if not self.self_player or not self.self_player.position:
            return []
            
        nearby = []
        for player in self.known_players.values():
            if player.id == "self":
                continue
            if relation and player.relation != relation:
                continue
            if player.position:
                dist = self.self_player.position.distance_to(player.position)
                if dist <= max_distance:
                    nearby.append(player)
                    
        return sorted(
            nearby,
            key=lambda p: self.self_player.position.distance_to(p.position)
        )
        
    def get_team_status(self, team_id: Optional[str] = None) -> Optional[TeamInfo]:
        """Get team information."""
        if team_id:
            return self.teams.get(team_id)
            
        # Return own team
        if self.self_player and self.self_player.team:
            return self.teams.get(self.self_player.team)
            
        return None
        
    def get_enemies(self) -> List[Player]:
        """Get all known enemy players."""
        return [
            p for p in self.known_players.values()
            if p.relation == PlayerRelation.ENEMY
        ]
        
    def get_allies(self) -> List[Player]:
        """Get all known allied players."""
        return [
            p for p in self.known_players.values()
            if p.is_friendly() and p.id != "self"
        ]
        
    def generate_callout(self, player: Player) -> str:
        """Generate voice callout for player."""
        direction = ""
        distance = ""
        
        if self.self_player and self.self_player.position and player.position:
            dist = self.self_player.position.distance_to(player.position)
            
            if dist < 20:
                distance = "close"
            elif dist < 50:
                distance = "medium range"
            else:
                distance = "far"
                
            # Calculate direction
            dx = player.position.x - self.self_player.position.x
            dy = player.position.y - self.self_player.position.y
            
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "ahead" if dy > 0 else "behind"
                
        relation = "Enemy" if player.is_hostile() else "Player"
        
        parts = [relation]
        if player.class_type:
            parts.append(player.class_type)
        if direction:
            parts.append(direction)
        if distance:
            parts.append(distance)
            
        return " ".join(parts)
        
    def save_state(self, filepath: str):
        """Save state to file."""
        state = {
            'known_players': {
                pid: {
                    'id': p.id,
                    'name': p.name,
                    'relation': p.relation.value,
                    'status': p.status.value,
                    'team': p.team,
                    'score': p.score
                }
                for pid, p in self.known_players.items()
            },
            'friends_list': list(self.friends_list),
            'party_members': list(self.party_members)
        }
        
        Path(filepath).write_text(json.dumps(state, indent=2))
        
    def load_state(self, filepath: str):
        """Load state from file."""
        path = Path(filepath)
        if not path.exists():
            return
            
        state = json.loads(path.read_text())
        
        self.friends_list = set(state.get('friends_list', []))
        self.party_members = set(state.get('party_members', []))
        
        for pid, pdata in state.get('known_players', {}).items():
            self.known_players[pid] = Player(
                id=pdata['id'],
                name=pdata['name'],
                relation=PlayerRelation(pdata.get('relation', 'unknown')),
                status=PlayerStatus(pdata.get('status', 'unknown')),
                team=pdata.get('team'),
                score=pdata.get('score', 0)
            )


# Convenience functions
def create_awareness(player_name: str, **kwargs) -> MultiplayerAwareness:
    """Create multiplayer awareness with self set."""
    mp = MultiplayerAwareness()
    mp.set_self(player_name, **kwargs)
    return mp


def detect_players_in_screenshot(screenshot: Any) -> List[Player]:
    """Quick player detection from screenshot."""
    detector = PlayerDetector()
    return detector.detect_from_screen(screenshot)
