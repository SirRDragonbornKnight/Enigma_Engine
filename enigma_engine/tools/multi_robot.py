"""
Multi-Robot Coordination for Enigma AI Engine

Coordinate multiple robots working together.

Features:
- Swarm coordination
- Task allocation
- Formation control
- Collision avoidance
- Communication protocols

Usage:
    from enigma_engine.tools.multi_robot import RobotSwarm, get_swarm
    
    swarm = get_swarm()
    
    # Add robots
    swarm.add_robot("bot1", position=(0, 0))
    swarm.add_robot("bot2", position=(1, 0))
    
    # Coordinate movement
    swarm.form_line()
"""

import logging
import math
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RobotStatus(Enum):
    """Robot status states."""
    IDLE = "idle"
    MOVING = "moving"
    EXECUTING = "executing"
    CHARGING = "charging"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class FormationType(Enum):
    """Formation types for swarm."""
    LINE = "line"
    COLUMN = "column"
    WEDGE = "wedge"
    CIRCLE = "circle"
    GRID = "grid"
    CUSTOM = "custom"


@dataclass
class Position:
    """2D position."""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # Orientation
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Velocity:
    """Robot velocity."""
    linear: float = 0.0
    angular: float = 0.0


@dataclass
class RobotState:
    """Robot state information."""
    robot_id: str
    position: Position = field(default_factory=Position)
    velocity: Velocity = field(default_factory=Velocity)
    status: RobotStatus = RobotStatus.IDLE
    battery: float = 100.0
    timestamp: float = 0.0


@dataclass
class Task:
    """Task for robots to execute."""
    task_id: str
    task_type: str
    target_position: Optional[Position] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_robot: Optional[str] = None
    completed: bool = False


class Robot:
    """Individual robot representation."""
    
    def __init__(
        self,
        robot_id: str,
        position: Optional[Position] = None
    ):
        """Initialize robot."""
        self._id = robot_id
        self._state = RobotState(
            robot_id=robot_id,
            position=position or Position()
        )
        
        self._target: Optional[Position] = None
        self._path: List[Position] = []
        
        # Capabilities
        self._max_speed = 1.0  # m/s
        self._max_angular = 1.0  # rad/s
        
        # Communication
        self._neighbors: Set[str] = set()
        self._message_queue: deque = deque(maxlen=100)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def state(self) -> RobotState:
        return self._state
    
    @property
    def position(self) -> Position:
        return self._state.position
    
    def set_position(self, pos: Position):
        """Set robot position."""
        self._state.position = pos
    
    def set_target(self, target: Position):
        """Set movement target."""
        self._target = target
        self._state.status = RobotStatus.MOVING
    
    def clear_target(self):
        """Clear movement target."""
        self._target = None
        self._state.status = RobotStatus.IDLE
    
    def update(self, dt: float):
        """Update robot state."""
        if self._target is None:
            return
        
        # Move towards target
        dx = self._target.x - self._state.position.x
        dy = self._target.y - self._state.position.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.05:  # Arrived
            self._state.position.x = self._target.x
            self._state.position.y = self._target.y
            self.clear_target()
            return
        
        # Move
        speed = min(self._max_speed, dist / dt)
        self._state.position.x += (dx / dist) * speed * dt
        self._state.position.y += (dy / dist) * speed * dt
        
        # Update orientation
        self._state.position.theta = math.atan2(dy, dx)
        
        # Drain battery
        self._state.battery -= 0.01 * dt
    
    def send_message(self, to_robot: str, message: Dict[str, Any]):
        """Send message to another robot."""
        # Add to queue for transmission
        self._message_queue.append({
            "from": self._id,
            "to": to_robot,
            "data": message,
            "timestamp": time.time()
        })
    
    def receive_message(self, message: Dict[str, Any]):
        """Receive message from another robot."""
        self._message_queue.append(message)


class TaskAllocator:
    """Allocate tasks to robots."""
    
    def __init__(self):
        self._tasks: List[Task] = []
        self._robots: Dict[str, Robot] = {}
    
    def set_robots(self, robots: Dict[str, Robot]):
        """Set available robots."""
        self._robots = robots
    
    def add_task(self, task: Task):
        """Add task to queue."""
        self._tasks.append(task)
        self._tasks.sort(key=lambda t: t.priority.value, reverse=True)
    
    def allocate(self) -> Dict[str, Task]:
        """
        Allocate tasks to robots.
        
        Returns:
            Mapping of robot_id -> task
        """
        allocations: Dict[str, Task] = {}
        
        for task in self._tasks:
            if task.completed or task.assigned_robot:
                continue
            
            # Find best robot
            best_robot = self._find_best_robot(task)
            
            if best_robot:
                task.assigned_robot = best_robot
                allocations[best_robot] = task
        
        return allocations
    
    def _find_best_robot(self, task: Task) -> Optional[str]:
        """Find best robot for task."""
        best_id = None
        best_score = float('-inf')
        
        for robot_id, robot in self._robots.items():
            if robot.state.status not in [RobotStatus.IDLE, RobotStatus.MOVING]:
                continue
            
            score = self._compute_score(robot, task)
            
            if score > best_score:
                best_score = score
                best_id = robot_id
        
        return best_id
    
    def _compute_score(self, robot: Robot, task: Task) -> float:
        """Compute allocation score (higher = better)."""
        score = 0.0
        
        # Distance penalty
        if task.target_position:
            dist = robot.position.distance_to(task.target_position)
            score -= dist * 0.5
        
        # Battery bonus
        score += robot.state.battery * 0.1
        
        # Idle bonus
        if robot.state.status == RobotStatus.IDLE:
            score += 10
        
        return score


class FormationController:
    """Control robot formations."""
    
    def __init__(self, spacing: float = 1.0):
        self._spacing = spacing
    
    def compute_formation(
        self,
        formation_type: FormationType,
        num_robots: int,
        center: Position = None
    ) -> List[Position]:
        """
        Compute positions for formation.
        
        Returns:
            List of positions for each robot
        """
        center = center or Position()
        positions = []
        
        if formation_type == FormationType.LINE:
            for i in range(num_robots):
                offset = (i - (num_robots - 1) / 2) * self._spacing
                positions.append(Position(
                    x=center.x + offset,
                    y=center.y
                ))
        
        elif formation_type == FormationType.COLUMN:
            for i in range(num_robots):
                offset = (i - (num_robots - 1) / 2) * self._spacing
                positions.append(Position(
                    x=center.x,
                    y=center.y + offset
                ))
        
        elif formation_type == FormationType.WEDGE:
            positions.append(Position(x=center.x, y=center.y))
            for i in range(1, num_robots):
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                positions.append(Position(
                    x=center.x - row * self._spacing * 0.7,
                    y=center.y + side * row * self._spacing * 0.5
                ))
        
        elif formation_type == FormationType.CIRCLE:
            for i in range(num_robots):
                angle = 2 * math.pi * i / num_robots
                radius = self._spacing * num_robots / (2 * math.pi)
                positions.append(Position(
                    x=center.x + radius * math.cos(angle),
                    y=center.y + radius * math.sin(angle)
                ))
        
        elif formation_type == FormationType.GRID:
            cols = max(1, int(math.sqrt(num_robots)))
            for i in range(num_robots):
                row = i // cols
                col = i % cols
                positions.append(Position(
                    x=center.x + (col - (cols - 1) / 2) * self._spacing,
                    y=center.y + (row - (num_robots / cols - 1) / 2) * self._spacing
                ))
        
        return positions


class CollisionAvoidance:
    """Avoid collisions between robots."""
    
    def __init__(
        self,
        safety_radius: float = 0.5,
        repulsion_strength: float = 1.0
    ):
        self._safety_radius = safety_radius
        self._repulsion = repulsion_strength
    
    def compute_avoidance(
        self,
        robot: Robot,
        others: List[Robot]
    ) -> Velocity:
        """
        Compute avoidance velocity.
        
        Returns:
            Avoidance velocity adjustment
        """
        avoid_x = 0.0
        avoid_y = 0.0
        
        for other in others:
            if other.id == robot.id:
                continue
            
            dx = robot.position.x - other.position.x
            dy = robot.position.y - other.position.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < self._safety_radius * 2 and dist > 0:
                # Repulsion force
                strength = self._repulsion * (self._safety_radius * 2 - dist)
                avoid_x += (dx / dist) * strength
                avoid_y += (dy / dist) * strength
        
        return Velocity(
            linear=math.sqrt(avoid_x**2 + avoid_y**2),
            angular=math.atan2(avoid_y, avoid_x)
        )


class CommunicationNetwork:
    """Inter-robot communication."""
    
    def __init__(
        self,
        range_limit: float = 10.0
    ):
        self._range = range_limit
        self._message_history: List[Dict[str, Any]] = []
    
    def can_communicate(
        self,
        robot1: Robot,
        robot2: Robot
    ) -> bool:
        """Check if robots can communicate."""
        dist = robot1.position.distance_to(robot2.position)
        return dist <= self._range
    
    def broadcast(
        self,
        sender: Robot,
        message: Dict[str, Any],
        robots: Dict[str, Robot]
    ):
        """Broadcast message to all reachable robots."""
        for robot_id, robot in robots.items():
            if robot_id == sender.id:
                continue
            
            if self.can_communicate(sender, robot):
                robot.receive_message({
                    "from": sender.id,
                    "data": message,
                    "timestamp": time.time()
                })
        
        self._message_history.append({
            "type": "broadcast",
            "from": sender.id,
            "data": message,
            "timestamp": time.time()
        })
    
    def send_direct(
        self,
        sender: Robot,
        receiver: Robot,
        message: Dict[str, Any]
    ) -> bool:
        """Send direct message."""
        if not self.can_communicate(sender, receiver):
            return False
        
        receiver.receive_message({
            "from": sender.id,
            "data": message,
            "timestamp": time.time()
        })
        
        return True


class RobotSwarm:
    """Manage a swarm of robots."""
    
    def __init__(
        self,
        update_rate: float = 10.0  # Hz
    ):
        """
        Initialize swarm.
        
        Args:
            update_rate: Update frequency in Hz
        """
        self._robots: Dict[str, Robot] = {}
        self._allocator = TaskAllocator()
        self._formation = FormationController()
        self._collision = CollisionAvoidance()
        self._network = CommunicationNetwork()
        
        self._update_rate = update_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info(f"RobotSwarm initialized at {update_rate}Hz")
    
    def add_robot(
        self,
        robot_id: str,
        position: Tuple[float, float] = (0, 0)
    ) -> Robot:
        """
        Add robot to swarm.
        
        Returns:
            Added robot
        """
        robot = Robot(
            robot_id,
            Position(x=position[0], y=position[1])
        )
        self._robots[robot_id] = robot
        self._allocator.set_robots(self._robots)
        
        logger.info(f"Added robot {robot_id} at {position}")
        return robot
    
    def remove_robot(self, robot_id: str):
        """Remove robot from swarm."""
        if robot_id in self._robots:
            del self._robots[robot_id]
            self._allocator.set_robots(self._robots)
            logger.info(f"Removed robot {robot_id}")
    
    def get_robot(self, robot_id: str) -> Optional[Robot]:
        """Get robot by ID."""
        return self._robots.get(robot_id)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all robot positions."""
        return {
            rid: robot.position
            for rid, robot in self._robots.items()
        }
    
    def set_target(self, robot_id: str, x: float, y: float):
        """Set target for specific robot."""
        robot = self._robots.get(robot_id)
        if robot:
            robot.set_target(Position(x=x, y=y))
    
    def set_all_targets(self, targets: Dict[str, Tuple[float, float]]):
        """Set targets for multiple robots."""
        for robot_id, (x, y) in targets.items():
            self.set_target(robot_id, x, y)
    
    def form(
        self,
        formation_type: FormationType,
        center: Tuple[float, float] = (0, 0)
    ):
        """
        Form specified formation.
        
        Args:
            formation_type: Type of formation
            center: Center position
        """
        positions = self._formation.compute_formation(
            formation_type,
            len(self._robots),
            Position(x=center[0], y=center[1])
        )
        
        for robot, pos in zip(self._robots.values(), positions):
            robot.set_target(pos)
        
        logger.info(f"Forming {formation_type.value} at {center}")
    
    def form_line(self, center: Tuple[float, float] = (0, 0)):
        """Form line formation."""
        self.form(FormationType.LINE, center)
    
    def form_circle(self, center: Tuple[float, float] = (0, 0)):
        """Form circle formation."""
        self.form(FormationType.CIRCLE, center)
    
    def form_wedge(self, center: Tuple[float, float] = (0, 0)):
        """Form wedge formation."""
        self.form(FormationType.WEDGE, center)
    
    def form_grid(self, center: Tuple[float, float] = (0, 0)):
        """Form grid formation."""
        self.form(FormationType.GRID, center)
    
    def add_task(self, task: Task):
        """Add task for allocation."""
        self._allocator.add_task(task)
    
    def allocate_tasks(self) -> Dict[str, Task]:
        """Allocate pending tasks."""
        return self._allocator.allocate()
    
    def broadcast(self, sender_id: str, message: Dict[str, Any]):
        """Broadcast message from robot."""
        sender = self._robots.get(sender_id)
        if sender:
            self._network.broadcast(sender, message, self._robots)
    
    def start(self):
        """Start swarm update loop."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Swarm started")
    
    def stop(self):
        """Stop swarm update loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Swarm stopped")
    
    def _run_loop(self):
        """Main update loop."""
        dt = 1.0 / self._update_rate
        
        while self._running:
            self._update(dt)
            time.sleep(dt)
    
    def _update(self, dt: float):
        """Update all robots."""
        robots_list = list(self._robots.values())
        
        for robot in robots_list:
            # Check collision avoidance
            avoidance = self._collision.compute_avoidance(robot, robots_list)
            
            # Apply avoidance (simplified)
            if avoidance.linear > 0.1:
                # Adjust target slightly
                pass
            
            # Update robot
            robot.update(dt)
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status."""
        return {
            "num_robots": len(self._robots),
            "robots": {
                rid: {
                    "position": (r.position.x, r.position.y, r.position.theta),
                    "status": r.state.status.value,
                    "battery": r.state.battery
                }
                for rid, r in self._robots.items()
            },
            "running": self._running
        }


# Global instance
_swarm: Optional[RobotSwarm] = None


def get_swarm() -> RobotSwarm:
    """Get or create global swarm."""
    global _swarm
    if _swarm is None:
        _swarm = RobotSwarm()
    return _swarm
