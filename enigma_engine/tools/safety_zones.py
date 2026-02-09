"""
Safety Zones for Enigma AI Engine

Define safety zones (no-go areas) for robots.

Features:
- Polygon zones
- Circular zones
- Zone types (forbidden, speed limit, warning)
- Real-time zone checking
- Path safety verification

Usage:
    from enigma_engine.tools.safety_zones import SafetyManager, get_safety
    
    safety = get_safety()
    
    # Add no-go zone
    safety.add_polygon_zone(
        "no_go_1",
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        ZoneType.FORBIDDEN
    )
    
    # Check if position is safe
    is_safe = safety.is_safe(0.5, 0.5)  # False - inside zone
"""

import logging
import math
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Types of safety zones."""
    FORBIDDEN = "forbidden"  # No entry
    SPEED_LIMIT = "speed_limit"  # Reduce speed
    WARNING = "warning"  # Warning only
    GEOFENCE = "geofence"  # Must stay inside
    TEMPORARY = "temporary"  # Dynamic obstacle


@dataclass
class Point2D:
    """2D point."""
    x: float
    y: float


@dataclass 
class Zone:
    """Base safety zone."""
    zone_id: str
    zone_type: ZoneType
    active: bool = True
    
    # Speed limit (for SPEED_LIMIT zones)
    speed_limit: float = 0.5  # m/s
    
    # Metadata
    description: str = ""
    created_at: float = 0.0


@dataclass
class PolygonZone(Zone):
    """Polygon safety zone."""
    vertices: List[Tuple[float, float]] = field(default_factory=list)
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside polygon (ray casting)."""
        n = len(self.vertices)
        if n < 3:
            return False
        
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)."""
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class CircleZone(Zone):
    """Circular safety zone."""
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 1.0
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside circle."""
        dx = x - self.center_x
        dy = y - self.center_y
        return (dx * dx + dy * dy) <= (self.radius * self.radius)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box."""
        return (
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius
        )


@dataclass
class RectangleZone(Zone):
    """Rectangular safety zone."""
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside rectangle."""
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box."""
        return (self.min_x, self.min_y, self.max_x, self.max_y)


@dataclass
class CheckResult:
    """Result of safety check."""
    safe: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_speed: Optional[float] = None


class SafetyManager:
    """Manage safety zones."""
    
    def __init__(self):
        """Initialize safety manager."""
        self._zones: Dict[str, Zone] = {}
        
        # Geofence (if set, robot must stay inside)
        self._geofence: Optional[Zone] = None
        
        # Callbacks
        self._violation_callbacks: List[Callable[[str, str], None]] = []
        
        logger.info("SafetyManager initialized")
    
    def add_polygon_zone(
        self,
        zone_id: str,
        vertices: List[Tuple[float, float]],
        zone_type: ZoneType = ZoneType.FORBIDDEN,
        **kwargs
    ) -> PolygonZone:
        """
        Add polygon zone.
        
        Args:
            zone_id: Unique identifier
            vertices: List of (x, y) vertices
            zone_type: Type of zone
            
        Returns:
            Created zone
        """
        zone = PolygonZone(
            zone_id=zone_id,
            zone_type=zone_type,
            vertices=list(vertices),
            **kwargs
        )
        self._zones[zone_id] = zone
        
        if zone_type == ZoneType.GEOFENCE:
            self._geofence = zone
        
        logger.info(f"Added polygon zone {zone_id} ({zone_type.value})")
        return zone
    
    def add_circle_zone(
        self,
        zone_id: str,
        center: Tuple[float, float],
        radius: float,
        zone_type: ZoneType = ZoneType.FORBIDDEN,
        **kwargs
    ) -> CircleZone:
        """
        Add circular zone.
        
        Args:
            zone_id: Unique identifier
            center: (x, y) center
            radius: Circle radius
            zone_type: Type of zone
            
        Returns:
            Created zone
        """
        zone = CircleZone(
            zone_id=zone_id,
            zone_type=zone_type,
            center_x=center[0],
            center_y=center[1],
            radius=radius,
            **kwargs
        )
        self._zones[zone_id] = zone
        
        logger.info(f"Added circle zone {zone_id} ({zone_type.value})")
        return zone
    
    def add_rectangle_zone(
        self,
        zone_id: str,
        min_corner: Tuple[float, float],
        max_corner: Tuple[float, float],
        zone_type: ZoneType = ZoneType.FORBIDDEN,
        **kwargs
    ) -> RectangleZone:
        """
        Add rectangular zone.
        
        Args:
            zone_id: Unique identifier
            min_corner: (min_x, min_y)
            max_corner: (max_x, max_y)
            zone_type: Type of zone
            
        Returns:
            Created zone
        """
        zone = RectangleZone(
            zone_id=zone_id,
            zone_type=zone_type,
            min_x=min_corner[0],
            min_y=min_corner[1],
            max_x=max_corner[0],
            max_y=max_corner[1],
            **kwargs
        )
        self._zones[zone_id] = zone
        
        logger.info(f"Added rectangle zone {zone_id} ({zone_type.value})")
        return zone
    
    def remove_zone(self, zone_id: str):
        """Remove a zone."""
        if zone_id in self._zones:
            zone = self._zones[zone_id]
            del self._zones[zone_id]
            
            if zone == self._geofence:
                self._geofence = None
            
            logger.info(f"Removed zone {zone_id}")
    
    def set_zone_active(self, zone_id: str, active: bool):
        """Enable/disable a zone."""
        if zone_id in self._zones:
            self._zones[zone_id].active = active
    
    def is_safe(self, x: float, y: float) -> bool:
        """
        Check if position is safe.
        
        Returns:
            True if safe
        """
        return self.check_position(x, y).safe
    
    def check_position(self, x: float, y: float) -> CheckResult:
        """
        Detailed safety check for position.
        
        Returns:
            CheckResult with violations/warnings
        """
        result = CheckResult(safe=True)
        
        # Check geofence first
        if self._geofence and self._geofence.active:
            if not self._geofence.contains(x, y):
                result.safe = False
                result.violations.append(f"Outside geofence: {self._geofence.zone_id}")
        
        # Check all zones
        for zone_id, zone in self._zones.items():
            if not zone.active:
                continue
            
            if zone.zone_type == ZoneType.GEOFENCE:
                continue  # Already checked
            
            if zone.contains(x, y):
                if zone.zone_type == ZoneType.FORBIDDEN:
                    result.safe = False
                    result.violations.append(f"Inside forbidden zone: {zone_id}")
                    self._notify_violation(zone_id, "position")
                
                elif zone.zone_type == ZoneType.SPEED_LIMIT:
                    if result.max_speed is None or zone.speed_limit < result.max_speed:
                        result.max_speed = zone.speed_limit
                
                elif zone.zone_type == ZoneType.WARNING:
                    result.warnings.append(f"Warning zone: {zone_id}")
                
                elif zone.zone_type == ZoneType.TEMPORARY:
                    result.safe = False
                    result.violations.append(f"Temporary obstacle: {zone_id}")
        
        return result
    
    def check_path(
        self,
        path: List[Tuple[float, float]],
        resolution: float = 0.1
    ) -> CheckResult:
        """
        Check if entire path is safe.
        
        Args:
            path: List of (x, y) waypoints
            resolution: Check interval in meters
            
        Returns:
            Combined CheckResult
        """
        result = CheckResult(safe=True)
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Check points along segment
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            steps = max(1, int(dist / resolution))
            
            for j in range(steps + 1):
                t = j / steps
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                
                point_result = self.check_position(x, y)
                
                if not point_result.safe:
                    result.safe = False
                    result.violations.extend(point_result.violations)
                
                result.warnings.extend(point_result.warnings)
                
                if point_result.max_speed is not None:
                    if result.max_speed is None or point_result.max_speed < result.max_speed:
                        result.max_speed = point_result.max_speed
        
        # Deduplicate
        result.violations = list(set(result.violations))
        result.warnings = list(set(result.warnings))
        
        return result
    
    def get_safe_speed(self, x: float, y: float) -> float:
        """
        Get maximum safe speed at position.
        
        Returns:
            Max speed in m/s (default: inf if no limit)
        """
        result = self.check_position(x, y)
        
        if not result.safe:
            return 0.0
        
        return result.max_speed if result.max_speed else float('inf')
    
    def _notify_violation(self, zone_id: str, violation_type: str):
        """Notify callbacks of safety violation."""
        for callback in self._violation_callbacks:
            try:
                callback(zone_id, violation_type)
            except Exception as e:
                logger.error(f"Violation callback error: {e}")
    
    def on_violation(self, callback: Callable[[str, str], None]):
        """Register callback for safety violations."""
        self._violation_callbacks.append(callback)
    
    def get_zones(self) -> List[Zone]:
        """Get all zones."""
        return list(self._zones.values())
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID."""
        return self._zones.get(zone_id)
    
    def clear_zones(self):
        """Remove all zones."""
        self._zones.clear()
        self._geofence = None
    
    def save_zones(self, filepath: str):
        """Save zones to file."""
        data = []
        
        for zone in self._zones.values():
            entry = {
                "zone_id": zone.zone_id,
                "zone_type": zone.zone_type.value,
                "active": zone.active,
                "speed_limit": zone.speed_limit,
                "description": zone.description
            }
            
            if isinstance(zone, PolygonZone):
                entry["shape"] = "polygon"
                entry["vertices"] = zone.vertices
            elif isinstance(zone, CircleZone):
                entry["shape"] = "circle"
                entry["center"] = [zone.center_x, zone.center_y]
                entry["radius"] = zone.radius
            elif isinstance(zone, RectangleZone):
                entry["shape"] = "rectangle"
                entry["min"] = [zone.min_x, zone.min_y]
                entry["max"] = [zone.max_x, zone.max_y]
            
            data.append(entry)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} zones to {filepath}")
    
    def load_zones(self, filepath: str):
        """Load zones from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            zone_type = ZoneType(entry.get("zone_type", "forbidden"))
            kwargs = {
                "speed_limit": entry.get("speed_limit", 0.5),
                "description": entry.get("description", ""),
                "active": entry.get("active", True)
            }
            
            shape = entry.get("shape", "polygon")
            
            if shape == "polygon":
                self.add_polygon_zone(
                    entry["zone_id"],
                    entry["vertices"],
                    zone_type,
                    **kwargs
                )
            elif shape == "circle":
                self.add_circle_zone(
                    entry["zone_id"],
                    tuple(entry["center"]),
                    entry["radius"],
                    zone_type,
                    **kwargs
                )
            elif shape == "rectangle":
                self.add_rectangle_zone(
                    entry["zone_id"],
                    tuple(entry["min"]),
                    tuple(entry["max"]),
                    zone_type,
                    **kwargs
                )
        
        logger.info(f"Loaded {len(data)} zones from {filepath}")


# Global instance
_safety: Optional[SafetyManager] = None


def get_safety() -> SafetyManager:
    """Get or create global safety manager."""
    global _safety
    if _safety is None:
        _safety = SafetyManager()
    return _safety
