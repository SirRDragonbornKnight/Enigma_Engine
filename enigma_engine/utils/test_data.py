"""
Test Data Generation - Generate test fixtures.

Provides data generation for testing:
- Random data generators
- Faker-like text generation
- Model/schema-based generation
- Deterministic seeded generation
- Batch generation

Part of the Enigma AI Engine testing utilities.
"""

import hashlib
import json
import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')


# Common data pools
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oliver", "Patricia",
    "Quinn", "Rachel", "Samuel", "Tina", "Ulysses", "Victoria", "William", "Xena",
    "Yusuf", "Zara", "Aiden", "Bella", "Caleb", "Daisy"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

DOMAINS = [
    "example.com", "test.org", "demo.net", "sample.io", "mock.dev",
    "fake.co", "testing.edu", "dummy.biz"
]

LOREM_WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
    "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
    "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
    "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
    "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
    "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
    "deserunt", "mollit", "anim", "id", "est", "laborum"
]

COMPANY_SUFFIXES = ["Inc", "LLC", "Corp", "Ltd", "Co", "Group", "Solutions", "Tech"]

STREET_SUFFIXES = ["Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Way", "Court"]


class Generator:
    """
    Test data generator with seeded randomness.
    
    Usage:
        gen = Generator(seed=42)  # Reproducible
        
        # Basic types
        name = gen.name()           # "Alice Smith"
        email = gen.email()         # "alice.smith@example.com"
        phone = gen.phone()         # "+1-555-123-4567"
        
        # Numbers
        age = gen.integer(18, 80)   # 45
        price = gen.decimal(10, 100, decimals=2)  # 54.67
        
        # Text
        word = gen.word()           # "ipsum"
        sentence = gen.sentence()   # "Lorem ipsum dolor sit amet."
        paragraph = gen.paragraph() # Multiple sentences
        
        # Dates
        date = gen.date()           # datetime object
        past = gen.past_date(days=30)
        future = gen.future_date(days=7)
        
        # Collections
        items = gen.list_of(gen.name, 5)  # 5 names
        
        # Custom
        status = gen.choice(["active", "pending", "inactive"])
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = random.Random(seed)
        self._counter = 0
    
    def reset(self, seed: Optional[int] = None):
        """Reset generator with optional new seed."""
        self._seed = seed if seed is not None else self._seed
        self._rng = random.Random(self._seed)
        self._counter = 0
    
    # === Identity ===
    
    def first_name(self) -> str:
        """Generate a first name."""
        return self._rng.choice(FIRST_NAMES)
    
    def last_name(self) -> str:
        """Generate a last name."""
        return self._rng.choice(LAST_NAMES)
    
    def name(self) -> str:
        """Generate a full name."""
        return f"{self.first_name()} {self.last_name()}"
    
    def username(self) -> str:
        """Generate a username."""
        return f"{self.first_name().lower()}{self._rng.randint(1, 999)}"
    
    def email(self, name: Optional[str] = None) -> str:
        """Generate an email address."""
        if name:
            local = name.lower().replace(" ", ".")
        else:
            local = f"{self.first_name().lower()}.{self.last_name().lower()}"
        domain = self._rng.choice(DOMAINS)
        return f"{local}@{domain}"
    
    def phone(self, format: str = "+1-###-###-####") -> str:
        """Generate a phone number."""
        result = ""
        for char in format:
            if char == "#":
                result += str(self._rng.randint(0, 9))
            else:
                result += char
        return result
    
    def company(self) -> str:
        """Generate a company name."""
        name = self.last_name()
        suffix = self._rng.choice(COMPANY_SUFFIXES)
        return f"{name} {suffix}"
    
    # === Numbers ===
    
    def integer(self, min_val: int = 0, max_val: int = 100) -> int:
        """Generate a random integer."""
        return self._rng.randint(min_val, max_val)
    
    def decimal(
        self,
        min_val: float = 0.0,
        max_val: float = 100.0,
        decimals: int = 2
    ) -> float:
        """Generate a random decimal number."""
        value = self._rng.uniform(min_val, max_val)
        return round(value, decimals)
    
    def boolean(self, true_probability: float = 0.5) -> bool:
        """Generate a random boolean."""
        return self._rng.random() < true_probability
    
    def percentage(self) -> int:
        """Generate a percentage (0-100)."""
        return self._rng.randint(0, 100)
    
    # === Text ===
    
    def word(self) -> str:
        """Generate a random word."""
        return self._rng.choice(LOREM_WORDS)
    
    def words(self, count: int = 5) -> str:
        """Generate multiple words."""
        return " ".join(self._rng.choice(LOREM_WORDS) for _ in range(count))
    
    def sentence(self, words: int = 8) -> str:
        """Generate a sentence."""
        text = self.words(words)
        return text.capitalize() + "."
    
    def sentences(self, count: int = 3, words_per: int = 8) -> str:
        """Generate multiple sentences."""
        return " ".join(self.sentence(words_per) for _ in range(count))
    
    def paragraph(self, sentences: int = 5) -> str:
        """Generate a paragraph."""
        return self.sentences(sentences)
    
    def paragraphs(self, count: int = 3) -> str:
        """Generate multiple paragraphs."""
        return "\n\n".join(self.paragraph() for _ in range(count))
    
    def text(self, length: int = 200) -> str:
        """Generate text of approximate length."""
        result = ""
        while len(result) < length:
            result += self.sentence() + " "
        return result[:length].strip()
    
    # === Identifiers ===
    
    def uuid(self) -> str:
        """Generate a UUID."""
        # Seeded UUID generation
        data = f"{self._seed}:{self._counter}".encode()
        self._counter += 1
        return str(uuid.UUID(hashlib.md5(data).hexdigest()))
    
    def sequential_id(self, prefix: str = "") -> str:
        """Generate a sequential ID."""
        self._counter += 1
        return f"{prefix}{self._counter}"
    
    def hex_string(self, length: int = 8) -> str:
        """Generate a random hex string."""
        return "".join(self._rng.choice("0123456789abcdef") for _ in range(length))
    
    def alphanumeric(self, length: int = 10) -> str:
        """Generate alphanumeric string."""
        chars = string.ascii_letters + string.digits
        return "".join(self._rng.choice(chars) for _ in range(length))
    
    # === Dates ===
    
    def date(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> datetime:
        """Generate a random date."""
        if start is None:
            start = datetime(2020, 1, 1)
        if end is None:
            end = datetime.now()
        
        delta = end - start
        random_days = self._rng.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    def past_date(self, days: int = 30) -> datetime:
        """Generate a date in the past."""
        return datetime.now() - timedelta(days=self._rng.randint(1, days))
    
    def future_date(self, days: int = 30) -> datetime:
        """Generate a date in the future."""
        return datetime.now() + timedelta(days=self._rng.randint(1, days))
    
    def timestamp(self) -> int:
        """Generate a Unix timestamp."""
        return int(self.date().timestamp())
    
    def iso_date(self) -> str:
        """Generate an ISO format date string."""
        return self.date().isoformat()
    
    # === Location ===
    
    def address(self) -> str:
        """Generate a street address."""
        number = self._rng.randint(1, 9999)
        street = f"{self.last_name()} {self._rng.choice(STREET_SUFFIXES)}"
        return f"{number} {street}"
    
    def city(self) -> str:
        """Generate a city name."""
        return f"{self.first_name()}ville"
    
    def zip_code(self) -> str:
        """Generate a ZIP code."""
        return f"{self._rng.randint(10000, 99999)}"
    
    def country(self) -> str:
        """Generate a country code."""
        return self._rng.choice(["US", "UK", "CA", "AU", "DE", "FR", "JP", "BR"])
    
    def latitude(self) -> float:
        """Generate a latitude."""
        return self.decimal(-90, 90, 6)
    
    def longitude(self) -> float:
        """Generate a longitude."""
        return self.decimal(-180, 180, 6)
    
    def coordinates(self) -> tuple:
        """Generate lat/long coordinates."""
        return (self.latitude(), self.longitude())
    
    # === Internet ===
    
    def url(self) -> str:
        """Generate a URL."""
        domain = self._rng.choice(DOMAINS)
        path = self.word()
        return f"https://{domain}/{path}"
    
    def ip_address(self) -> str:
        """Generate an IPv4 address."""
        return ".".join(str(self._rng.randint(0, 255)) for _ in range(4))
    
    def mac_address(self) -> str:
        """Generate a MAC address."""
        return ":".join(f"{self._rng.randint(0, 255):02x}" for _ in range(6))
    
    def user_agent(self) -> str:
        """Generate a user agent string."""
        browsers = ["Chrome", "Firefox", "Safari", "Edge"]
        versions = ["100.0", "99.0", "98.0", "97.0"]
        return f"Mozilla/5.0 ({self._rng.choice(browsers)}/{self._rng.choice(versions)})"
    
    # === Collections ===
    
    def choice(self, options: list[T]) -> T:
        """Choose a random item from a list."""
        return self._rng.choice(options)
    
    def choices(self, options: list[T], count: int = 3) -> list[T]:
        """Choose multiple random items."""
        return self._rng.choices(options, k=count)
    
    def sample(self, options: list[T], count: int = 3) -> list[T]:
        """Choose unique random items."""
        return self._rng.sample(options, min(count, len(options)))
    
    def shuffle(self, items: list[T]) -> list[T]:
        """Shuffle a list."""
        result = items.copy()
        self._rng.shuffle(result)
        return result
    
    def list_of(
        self,
        generator: Callable[[], T],
        count: int = 5
    ) -> list[T]:
        """Generate a list using a generator function."""
        return [generator() for _ in range(count)]
    
    def dict_of(
        self,
        key_gen: Callable[[], str],
        value_gen: Callable[[], Any],
        count: int = 5
    ) -> dict[str, Any]:
        """Generate a dictionary."""
        return {key_gen(): value_gen() for _ in range(count)}
    
    # === JSON ===
    
    def json_object(self, depth: int = 2) -> dict[str, Any]:
        """Generate a random JSON-like object."""
        obj = {}
        for _ in range(self._rng.randint(2, 5)):
            key = self.word()
            if depth > 0 and self._rng.random() < 0.3:
                obj[key] = self.json_object(depth - 1)
            elif self._rng.random() < 0.5:
                obj[key] = self.sentence()
            else:
                obj[key] = self.integer()
        return obj
    
    def json_string(self) -> str:
        """Generate a JSON string."""
        return json.dumps(self.json_object())


@dataclass
class FieldSpec:
    """Specification for generating a field."""
    name: str
    generator: str  # Generator method name
    args: tuple = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)


class SchemaGenerator:
    """
    Generate data based on a schema.
    
    Usage:
        schema = SchemaGenerator(seed=42)
        
        # Define schema
        schema.define("user", [
            FieldSpec("id", "uuid"),
            FieldSpec("name", "name"),
            FieldSpec("email", "email"),
            FieldSpec("age", "integer", (18, 80)),
            FieldSpec("created_at", "iso_date"),
            FieldSpec("active", "boolean"),
        ])
        
        # Generate
        user = schema.generate("user")
        users = schema.generate_many("user", 10)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize schema generator."""
        self._generator = Generator(seed)
        self._schemas: dict[str, list[FieldSpec]] = {}
    
    def define(self, name: str, fields: list[FieldSpec]):
        """Define a schema."""
        self._schemas[name] = fields
    
    def generate(self, schema_name: str) -> dict[str, Any]:
        """Generate a single object from schema."""
        if schema_name not in self._schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        result = {}
        for field_spec in self._schemas[schema_name]:
            method = getattr(self._generator, field_spec.generator)
            result[field_spec.name] = method(*field_spec.args, **field_spec.kwargs)
        
        return result
    
    def generate_many(
        self,
        schema_name: str,
        count: int = 10
    ) -> list[dict[str, Any]]:
        """Generate multiple objects from schema."""
        return [self.generate(schema_name) for _ in range(count)]
    
    def to_json(self, schema_name: str, count: int = 1) -> str:
        """Generate schema data as JSON."""
        if count == 1:
            return json.dumps(self.generate(schema_name), indent=2, default=str)
        return json.dumps(self.generate_many(schema_name, count), indent=2, default=str)
    
    def to_file(
        self,
        schema_name: str,
        path: str,
        count: int = 10
    ):
        """Save generated data to a file."""
        data = self.generate_many(schema_name, count)
        Path(path).write_text(json.dumps(data, indent=2, default=str))


class FixtureFactory:
    """
    Factory for creating test fixtures.
    
    Usage:
        factory = FixtureFactory()
        
        # Register builders
        @factory.register("user")
        def build_user(gen, **overrides):
            return {
                "id": gen.uuid(),
                "name": gen.name(),
                "email": gen.email(),
                **overrides
            }
        
        # Create fixtures
        user = factory.create("user")
        admin = factory.create("user", role="admin")
        users = factory.create_batch("user", 5)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize fixture factory."""
        self._generator = Generator(seed)
        self._builders: dict[str, Callable] = {}
    
    def register(self, name: str):
        """Decorator to register a builder."""
        def decorator(func: Callable):
            self._builders[name] = func
            return func
        return decorator
    
    def create(self, name: str, **overrides) -> dict[str, Any]:
        """Create a fixture."""
        if name not in self._builders:
            raise ValueError(f"Unknown fixture: {name}")
        return self._builders[name](self._generator, **overrides)
    
    def create_batch(
        self,
        name: str,
        count: int = 5,
        **overrides
    ) -> list[dict[str, Any]]:
        """Create multiple fixtures."""
        return [self.create(name, **overrides) for _ in range(count)]


# Global generator instance
_global_generator: Optional[Generator] = None


def get_generator(seed: Optional[int] = None) -> Generator:
    """Get the global generator."""
    global _global_generator
    if _global_generator is None or seed is not None:
        _global_generator = Generator(seed)
    return _global_generator


# Convenience functions
def fake_name() -> str:
    """Generate a fake name."""
    return get_generator().name()


def fake_email() -> str:
    """Generate a fake email."""
    return get_generator().email()


def fake_text(length: int = 200) -> str:
    """Generate fake text."""
    return get_generator().text(length)


def fake_integer(min_val: int = 0, max_val: int = 100) -> int:
    """Generate a fake integer."""
    return get_generator().integer(min_val, max_val)


def fake_uuid() -> str:
    """Generate a fake UUID."""
    return get_generator().uuid()
