"""
Physics-Based Hair and Cloth Simulation

Simple physics simulation for avatar hair and clothing dynamics.
Uses spring-damper systems and collision detection.

FILE: enigma_engine/avatar/physics_simulation.py
TYPE: Avatar Animation
MAIN CLASSES: PhysicsSimulator, HairSimulator, ClothSimulator
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Vector3:
    """3D vector with physics operations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __neg__(self) -> 'Vector3':
        return Vector3(-self.x, -self.y, -self.z)
    
    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def length_squared(self) -> float:
        return self.x**2 + self.y**2 + self.z**2
    
    def normalized(self) -> 'Vector3':
        length = self.length()
        if length < 0.0001:
            return Vector3()
        return self * (1.0 / length)
    
    def lerp(self, target: 'Vector3', t: float) -> 'Vector3':
        return self + (target - self) * t


@dataclass
class PhysicsParticle:
    """A particle in the physics simulation."""
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    force: Vector3 = field(default_factory=Vector3)
    mass: float = 1.0
    is_pinned: bool = False  # Fixed in place
    damping: float = 0.98
    
    def apply_force(self, force: Vector3):
        """Apply a force to the particle."""
        self.force = self.force + force
    
    def integrate(self, dt: float):
        """Update position and velocity using Verlet integration."""
        if self.is_pinned:
            self.velocity = Vector3()
            return
        
        # Calculate acceleration
        self.acceleration = self.force * (1.0 / self.mass)
        
        # Update velocity with damping
        self.velocity = (self.velocity + self.acceleration * dt) * self.damping
        
        # Update position
        self.position = self.position + self.velocity * dt
        
        # Clear forces
        self.force = Vector3()


@dataclass
class Spring:
    """A spring connecting two particles."""
    particle_a: int  # Index of first particle
    particle_b: int  # Index of second particle
    rest_length: float
    stiffness: float = 500.0
    damping: float = 5.0
    
    def apply_force(self, particles: list[PhysicsParticle]):
        """Apply spring force to connected particles."""
        a = particles[self.particle_a]
        b = particles[self.particle_b]
        
        # Direction from a to b
        delta = b.position - a.position
        distance = delta.length()
        
        if distance < 0.0001:
            return
        
        # Normalize
        direction = delta * (1.0 / distance)
        
        # Spring force (Hooke's law)
        displacement = distance - self.rest_length
        spring_force = direction * (self.stiffness * displacement)
        
        # Damping force
        relative_velocity = b.velocity - a.velocity
        damping_force = direction * (relative_velocity.dot(direction) * self.damping)
        
        # Total force
        total_force = spring_force + damping_force
        
        # Apply to particles
        if not a.is_pinned:
            a.apply_force(total_force)
        if not b.is_pinned:
            b.apply_force(-total_force)


@dataclass
class Collider:
    """A collision object."""
    position: Vector3
    radius: float = 0.1
    
    def check_collision(self, particle: PhysicsParticle) -> bool:
        """Check if particle collides with this collider."""
        delta = particle.position - self.position
        return delta.length() < self.radius
    
    def resolve_collision(self, particle: PhysicsParticle):
        """Push particle out of collision."""
        delta = particle.position - self.position
        distance = delta.length()
        
        if distance < self.radius and distance > 0.0001:
            # Push out
            direction = delta.normalized()
            particle.position = self.position + direction * self.radius
            
            # Remove velocity into collider
            velocity_into = particle.velocity.dot(direction)
            if velocity_into < 0:
                particle.velocity = particle.velocity - direction * velocity_into * 1.5


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    gravity: Vector3 = field(default_factory=lambda: Vector3(0, -9.81, 0))
    wind: Vector3 = field(default_factory=Vector3)
    time_step: float = 0.016  # ~60 FPS
    substeps: int = 4
    air_resistance: float = 0.1
    floor_y: float = 0.0
    enable_floor: bool = True


class PhysicsSimulator:
    """Base physics simulator."""
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize simulator.
        
        Args:
            config: Physics configuration
        """
        self._config = config or PhysicsConfig()
        self._particles: list[PhysicsParticle] = []
        self._springs: list[Spring] = []
        self._colliders: list[Collider] = []
        self._last_update = time.time()
    
    def add_particle(self, particle: PhysicsParticle) -> int:
        """Add a particle, returns index."""
        self._particles.append(particle)
        return len(self._particles) - 1
    
    def add_spring(self, spring: Spring):
        """Add a spring constraint."""
        self._springs.append(spring)
    
    def add_collider(self, collider: Collider):
        """Add a collision object."""
        self._colliders.append(collider)
    
    def update(self, dt: Optional[float] = None):
        """
        Update physics simulation.
        
        Args:
            dt: Time delta (uses config time_step if None)
        """
        if dt is None:
            current = time.time()
            dt = min(current - self._last_update, 0.1)  # Cap at 100ms
            self._last_update = current
        
        # Substeps for stability
        substep_dt = dt / self._config.substeps
        
        for _ in range(self._config.substeps):
            self._substep(substep_dt)
    
    def _substep(self, dt: float):
        """Perform a single substep."""
        # Apply external forces
        for particle in self._particles:
            if particle.is_pinned:
                continue
            
            # Gravity
            particle.apply_force(self._config.gravity * particle.mass)
            
            # Wind
            particle.apply_force(self._config.wind)
            
            # Air resistance
            particle.apply_force(-particle.velocity * self._config.air_resistance)
        
        # Apply spring forces
        for spring in self._springs:
            spring.apply_force(self._particles)
        
        # Integrate
        for particle in self._particles:
            particle.integrate(dt)
        
        # Collision detection and response
        for particle in self._particles:
            if particle.is_pinned:
                continue
            
            # Floor collision
            if self._config.enable_floor and particle.position.y < self._config.floor_y:
                particle.position.y = self._config.floor_y
                if particle.velocity.y < 0:
                    particle.velocity.y *= -0.5  # Bounce
            
            # Collider collision
            for collider in self._colliders:
                if collider.check_collision(particle):
                    collider.resolve_collision(particle)
    
    def get_positions(self) -> list[Vector3]:
        """Get all particle positions."""
        return [p.position for p in self._particles]
    
    def reset(self):
        """Reset simulation."""
        self._particles.clear()
        self._springs.clear()


class HairSimulator(PhysicsSimulator):
    """Simulates hair strands as particle chains."""
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self._strands: list[list[int]] = []  # Lists of particle indices per strand
    
    def create_strand(self,
                      root_position: Vector3,
                      length: float,
                      segments: int = 5,
                      stiffness: float = 800.0,
                      mass_per_segment: float = 0.05) -> list[int]:
        """
        Create a hair strand.
        
        Args:
            root_position: Position of the root (attached to head)
            length: Total length of strand
            segments: Number of segments
            stiffness: Spring stiffness
            mass_per_segment: Mass of each segment
            
        Returns:
            List of particle indices
        """
        segment_length = length / segments
        indices = []
        
        for i in range(segments + 1):
            position = Vector3(
                root_position.x,
                root_position.y - i * segment_length,
                root_position.z
            )
            
            particle = PhysicsParticle(
                position=position,
                mass=mass_per_segment,
                is_pinned=(i == 0),  # Root is pinned
                damping=0.95
            )
            
            idx = self.add_particle(particle)
            indices.append(idx)
            
            # Add spring to previous particle
            if i > 0:
                spring = Spring(
                    particle_a=indices[i - 1],
                    particle_b=idx,
                    rest_length=segment_length,
                    stiffness=stiffness,
                    damping=10.0
                )
                self.add_spring(spring)
        
        self._strands.append(indices)
        return indices
    
    def move_root(self, strand_index: int, new_position: Vector3):
        """Move the root of a strand (follows head movement)."""
        if 0 <= strand_index < len(self._strands):
            root_idx = self._strands[strand_index][0]
            self._particles[root_idx].position = new_position
    
    def get_strand_positions(self, strand_index: int) -> list[Vector3]:
        """Get positions of all particles in a strand."""
        if 0 <= strand_index < len(self._strands):
            return [self._particles[i].position for i in self._strands[strand_index]]
        return []


class ClothSimulator(PhysicsSimulator):
    """Simulates cloth as a particle grid."""
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        super().__init__(config)
        self._width = 0
        self._height = 0
    
    def create_cloth(self,
                     top_left: Vector3,
                     width: float,
                     height: float,
                     width_segments: int = 10,
                     height_segments: int = 10,
                     stiffness: float = 500.0,
                     mass: float = 0.1) -> tuple[int, int]:
        """
        Create a cloth grid.
        
        Args:
            top_left: Top-left corner position
            width: Cloth width
            height: Cloth height
            width_segments: Horizontal segments
            height_segments: Vertical segments
            stiffness: Spring stiffness
            mass: Mass per particle
            
        Returns:
            Tuple of (width_particles, height_particles)
        """
        self._width = width_segments + 1
        self._height = height_segments + 1
        
        dx = width / width_segments
        dy = height / height_segments
        
        # Create particles
        indices = []
        for y in range(self._height):
            row = []
            for x in range(self._width):
                position = Vector3(
                    top_left.x + x * dx,
                    top_left.y - y * dy,
                    top_left.z
                )
                
                particle = PhysicsParticle(
                    position=position,
                    mass=mass,
                    is_pinned=(y == 0),  # Top row pinned
                    damping=0.97
                )
                
                idx = self.add_particle(particle)
                row.append(idx)
            indices.append(row)
        
        # Create springs
        for y in range(self._height):
            for x in range(self._width):
                idx = indices[y][x]
                
                # Structural springs (horizontal)
                if x < self._width - 1:
                    self.add_spring(Spring(
                        particle_a=idx,
                        particle_b=indices[y][x + 1],
                        rest_length=dx,
                        stiffness=stiffness
                    ))
                
                # Structural springs (vertical)
                if y < self._height - 1:
                    self.add_spring(Spring(
                        particle_a=idx,
                        particle_b=indices[y + 1][x],
                        rest_length=dy,
                        stiffness=stiffness
                    ))
                
                # Shear springs (diagonal)
                if x < self._width - 1 and y < self._height - 1:
                    diag = math.sqrt(dx**2 + dy**2)
                    self.add_spring(Spring(
                        particle_a=idx,
                        particle_b=indices[y + 1][x + 1],
                        rest_length=diag,
                        stiffness=stiffness * 0.5
                    ))
                    self.add_spring(Spring(
                        particle_a=indices[y][x + 1],
                        particle_b=indices[y + 1][x],
                        rest_length=diag,
                        stiffness=stiffness * 0.5
                    ))
                
                # Bend springs (skip one)
                if x < self._width - 2:
                    self.add_spring(Spring(
                        particle_a=idx,
                        particle_b=indices[y][x + 2],
                        rest_length=dx * 2,
                        stiffness=stiffness * 0.3
                    ))
                if y < self._height - 2:
                    self.add_spring(Spring(
                        particle_a=idx,
                        particle_b=indices[y + 2][x],
                        rest_length=dy * 2,
                        stiffness=stiffness * 0.3
                    ))
        
        return (self._width, self._height)
    
    def get_particle_at(self, x: int, y: int) -> Optional[PhysicsParticle]:
        """Get particle at grid position."""
        idx = y * self._width + x
        if 0 <= idx < len(self._particles):
            return self._particles[idx]
        return None
    
    def pin_corners(self):
        """Pin the top corners of the cloth."""
        if self._width > 0 and self._height > 0:
            self._particles[0].is_pinned = True  # Top-left
            self._particles[self._width - 1].is_pinned = True  # Top-right


# Factory functions
def create_hair_simulator(config: PhysicsConfig = None) -> HairSimulator:
    """Create a hair physics simulator."""
    return HairSimulator(config)


def create_cloth_simulator(config: PhysicsConfig = None) -> ClothSimulator:
    """Create a cloth physics simulator."""
    return ClothSimulator(config)


__all__ = [
    'PhysicsSimulator',
    'HairSimulator',
    'ClothSimulator',
    'PhysicsParticle',
    'PhysicsConfig',
    'Spring',
    'Collider',
    'Vector3',
    'create_hair_simulator',
    'create_cloth_simulator'
]
