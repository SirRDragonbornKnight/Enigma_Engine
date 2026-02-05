"""
Database Abstraction - Multi-backend database support.

Provides unified database interface for:
- SQLite (local)
- PostgreSQL
- MySQL
- In-memory
- MongoDB (NoSQL)

Part of the ForgeAI persistence utilities.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Iterator
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


@dataclass
class QueryResult:
    """Result from a database query."""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    affected_rows: int = 0
    last_insert_id: Optional[Union[int, str]] = None  # str for MongoDB ObjectId
    success: bool = True
    error: Optional[str] = None
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)
    
    def __len__(self) -> int:
        return len(self.rows)
    
    def first(self) -> Optional[Dict[str, Any]]:
        """Get first row."""
        return self.rows[0] if self.rows else None
    
    def scalar(self, column: Optional[str] = None) -> Any:
        """Get single value from first row."""
        if not self.rows:
            return None
        row = self.rows[0]
        if column:
            return row.get(column)
        return list(row.values())[0] if row else None


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to database."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close database connection."""
        pass
    
    @abstractmethod
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryResult:
        """Execute a query."""
        pass
    
    @abstractmethod
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> QueryResult:
        """Execute query with multiple parameter sets."""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> bool:
        """Start a transaction."""
        pass
    
    @abstractmethod
    def commit(self) -> bool:
        """Commit current transaction."""
        pass
    
    @abstractmethod
    def rollback(self) -> bool:
        """Rollback current transaction."""
        pass
    
    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        pass
    
    @abstractmethod
    def get_tables(self) -> List[str]:
        """List all tables."""
        pass
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        self.begin_transaction()
        try:
            yield self
            self.commit()
        except Exception:
            self.rollback()
            raise


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend."""
    
    def __init__(
        self,
        database: str = ":memory:",
        timeout: float = 30.0,
        check_same_thread: bool = False
    ):
        """
        Initialize SQLite backend.
        
        Args:
            database: Database file path or ":memory:"
            timeout: Connection timeout in seconds
            check_same_thread: Whether to check thread safety
        """
        self._database = database
        self._timeout = timeout
        self._check_same_thread = check_same_thread
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            if self._database != ":memory:":
                Path(self._database).parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = sqlite3.connect(
                self._database,
                timeout=self._timeout,
                check_same_thread=self._check_same_thread
            )
            self._connection.row_factory = sqlite3.Row
            return True
        except Exception:
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                return True
            except Exception:
                return False
        return True
    
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryResult:
        """Execute a query."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            with self._lock:
                cursor = self._connection.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Fetch results for SELECT queries
                rows = []
                if cursor.description:
                    columns = [col[0] for col in cursor.description]
                    for row in cursor.fetchall():
                        rows.append(dict(zip(columns, row)))
                
                return QueryResult(
                    rows=rows,
                    affected_rows=cursor.rowcount,
                    last_insert_id=cursor.lastrowid
                )
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> QueryResult:
        """Execute query with multiple parameter sets."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            with self._lock:
                cursor = self._connection.cursor()
                cursor.executemany(query, params_list)
                return QueryResult(
                    affected_rows=cursor.rowcount,
                    last_insert_id=cursor.lastrowid
                )
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def begin_transaction(self) -> bool:
        """Start a transaction."""
        if self._connection:
            try:
                self._connection.execute("BEGIN")
                return True
            except Exception:
                return False
        return False
    
    def commit(self) -> bool:
        """Commit current transaction."""
        if self._connection:
            try:
                self._connection.commit()
                return True
            except Exception:
                return False
        return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        if self._connection:
            try:
                self._connection.rollback()
                return True
            except Exception:
                return False
        return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        result = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return len(result.rows) > 0
    
    def get_tables(self) -> List[str]:
        """List all tables."""
        result = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row['name'] for row in result.rows]


class MemoryBackend(SQLiteBackend):
    """In-memory database backend (SQLite-based)."""
    
    def __init__(self):
        """Initialize in-memory database."""
        super().__init__(database=":memory:", check_same_thread=False)


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL database backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "forgeai",
        user: str = "postgres",
        password: str = ""
    ):
        """
        Initialize PostgreSQL backend.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._connection = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to PostgreSQL database."""
        try:
            import psycopg2
            import psycopg2.extras
            
            self._connection = psycopg2.connect(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                return True
            except Exception:
                return False
        return True
    
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryResult:
        """Execute a query."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            import psycopg2.extras
            
            with self._lock:
                cursor = self._connection.cursor(
                    cursor_factory=psycopg2.extras.RealDictCursor
                )
                cursor.execute(query, params)
                
                rows = []
                if cursor.description:
                    rows = [dict(row) for row in cursor.fetchall()]
                
                return QueryResult(
                    rows=rows,
                    affected_rows=cursor.rowcount
                )
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> QueryResult:
        """Execute query with multiple parameter sets."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            with self._lock:
                cursor = self._connection.cursor()
                cursor.executemany(query, params_list)
                return QueryResult(affected_rows=cursor.rowcount)
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def begin_transaction(self) -> bool:
        """Start a transaction."""
        # PostgreSQL auto-starts transactions
        return True
    
    def commit(self) -> bool:
        """Commit current transaction."""
        if self._connection:
            try:
                self._connection.commit()
                return True
            except Exception:
                return False
        return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        if self._connection:
            try:
                self._connection.rollback()
                return True
            except Exception:
                return False
        return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        result = self.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
            (table_name,)
        )
        return result.scalar() if result.success else False
    
    def get_tables(self) -> List[str]:
        """List all tables."""
        result = self.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        return [row['table_name'] for row in result.rows]


class MySQLBackend(DatabaseBackend):
    """MySQL database backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "forgeai",
        user: str = "root",
        password: str = ""
    ):
        """
        Initialize MySQL backend.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._connection = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to MySQL database."""
        try:
            import mysql.connector
            
            self._connection = mysql.connector.connect(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from database."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                return True
            except Exception:
                return False
        return True
    
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryResult:
        """Execute a query."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            with self._lock:
                cursor = self._connection.cursor(dictionary=True)
                cursor.execute(query, params)
                
                rows = []
                if cursor.description:
                    rows = cursor.fetchall()
                
                return QueryResult(
                    rows=rows,
                    affected_rows=cursor.rowcount,
                    last_insert_id=cursor.lastrowid
                )
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> QueryResult:
        """Execute query with multiple parameter sets."""
        if not self._connection:
            return QueryResult(success=False, error="Not connected")
        
        try:
            with self._lock:
                cursor = self._connection.cursor()
                cursor.executemany(query, params_list)
                return QueryResult(
                    affected_rows=cursor.rowcount,
                    last_insert_id=cursor.lastrowid
                )
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def begin_transaction(self) -> bool:
        """Start a transaction."""
        if self._connection:
            try:
                self._connection.start_transaction()
                return True
            except Exception:
                return False
        return False
    
    def commit(self) -> bool:
        """Commit current transaction."""
        if self._connection:
            try:
                self._connection.commit()
                return True
            except Exception:
                return False
        return False
    
    def rollback(self) -> bool:
        """Rollback current transaction."""
        if self._connection:
            try:
                self._connection.rollback()
                return True
            except Exception:
                return False
        return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        result = self.execute(
            "SELECT COUNT(*) as cnt FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
            (self._database, table_name)
        )
        return result.scalar('cnt') > 0 if result.success else False
    
    def get_tables(self) -> List[str]:
        """List all tables."""
        result = self.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (self._database,)
        )
        return [row['table_name'] for row in result.rows]


class MongoDBBackend(DatabaseBackend):
    """MongoDB database backend."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "forgeai",
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize MongoDB backend.
        
        Args:
            host: MongoDB host
            port: MongoDB port
            database: Database name
            username: Optional username
            password: Optional password
        """
        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password
        self._client = None
        self._db = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
            
            if self._username and self._password:
                uri = f"mongodb://{self._username}:{self._password}@{self._host}:{self._port}"
            else:
                uri = f"mongodb://{self._host}:{self._port}"
            
            self._client = MongoClient(uri)
            self._db = self._client[self._database]
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from MongoDB."""
        if self._client:
            try:
                self._client.close()
                self._client = None
                self._db = None
                return True
            except Exception:
                return False
        return True
    
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> QueryResult:
        """
        Execute a MongoDB operation.
        
        Query format: "collection.operation" (e.g., "users.find")
        Params: tuple with (filter_dict, [options_dict])
        """
        if not self._db:
            return QueryResult(success=False, error="Not connected")
        
        try:
            parts = query.split(".", 1)
            if len(parts) != 2:
                return QueryResult(success=False, error="Invalid query format")
            
            collection_name, operation = parts
            collection = self._db[collection_name]
            
            filter_dict = params[0] if params else {}
            options = params[1] if params and len(params) > 1 else {}
            
            with self._lock:
                if operation == "find":
                    cursor = collection.find(filter_dict, **options)
                    rows = [doc for doc in cursor]
                    # Convert ObjectId to string
                    for row in rows:
                        if '_id' in row:
                            row['_id'] = str(row['_id'])
                    return QueryResult(rows=rows)
                
                elif operation == "find_one":
                    doc = collection.find_one(filter_dict)
                    if doc:
                        doc['_id'] = str(doc['_id'])
                        return QueryResult(rows=[doc])
                    return QueryResult(rows=[])
                
                elif operation == "insert_one":
                    result = collection.insert_one(filter_dict)
                    return QueryResult(
                        last_insert_id=str(result.inserted_id),
                        affected_rows=1
                    )
                
                elif operation == "insert_many":
                    result = collection.insert_many(filter_dict)
                    return QueryResult(affected_rows=len(result.inserted_ids))
                
                elif operation == "update_one":
                    update = options.get('update', {})
                    result = collection.update_one(filter_dict, update)
                    return QueryResult(affected_rows=result.modified_count)
                
                elif operation == "update_many":
                    update = options.get('update', {})
                    result = collection.update_many(filter_dict, update)
                    return QueryResult(affected_rows=result.modified_count)
                
                elif operation == "delete_one":
                    result = collection.delete_one(filter_dict)
                    return QueryResult(affected_rows=result.deleted_count)
                
                elif operation == "delete_many":
                    result = collection.delete_many(filter_dict)
                    return QueryResult(affected_rows=result.deleted_count)
                
                elif operation == "count":
                    count = collection.count_documents(filter_dict)
                    return QueryResult(rows=[{"count": count}])
                
                else:
                    return QueryResult(success=False, error=f"Unknown operation: {operation}")
                    
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def execute_many(
        self,
        query: str,
        params_list: List[Tuple]
    ) -> QueryResult:
        """Execute multiple MongoDB operations."""
        total_affected = 0
        for params in params_list:
            result = self.execute(query, params)
            if result.success:
                total_affected += result.affected_rows
        return QueryResult(affected_rows=total_affected)
    
    def begin_transaction(self) -> bool:
        """MongoDB transactions require replica set."""
        return True
    
    def commit(self) -> bool:
        """MongoDB auto-commits."""
        return True
    
    def rollback(self) -> bool:
        """MongoDB transactions require replica set."""
        return True
    
    def table_exists(self, table_name: str) -> bool:
        """Check if collection exists."""
        if self._db:
            return table_name in self._db.list_collection_names()
        return False
    
    def get_tables(self) -> List[str]:
        """List all collections."""
        if self._db:
            return self._db.list_collection_names()
        return []


class DatabaseManager:
    """
    Manage multiple database connections.
    
    Usage:
        manager = DatabaseManager()
        
        # Register databases
        manager.register("main", SQLiteBackend("data/main.db"))
        manager.register("cache", MemoryBackend())
        
        # Connect
        manager.connect_all()
        
        # Use
        result = manager.get("main").execute("SELECT * FROM users")
        
        # With context manager
        with manager.get("main").transaction():
            manager.get("main").execute("INSERT INTO users VALUES (?)", ("alice",))
    """
    
    def __init__(self):
        """Initialize database manager."""
        self._databases: Dict[str, DatabaseBackend] = {}
        self._default: Optional[str] = None
    
    def register(
        self,
        name: str,
        backend: DatabaseBackend,
        default: bool = False
    ) -> None:
        """
        Register a database backend.
        
        Args:
            name: Database identifier
            backend: Database backend instance
            default: Set as default database
        """
        self._databases[name] = backend
        if default or self._default is None:
            self._default = name
    
    def unregister(self, name: str) -> bool:
        """Remove a database registration."""
        if name in self._databases:
            self._databases[name].disconnect()
            del self._databases[name]
            if self._default == name:
                self._default = next(iter(self._databases), None)
            return True
        return False
    
    def get(self, name: Optional[str] = None) -> Optional[DatabaseBackend]:
        """
        Get a database backend.
        
        Args:
            name: Database name (default if None)
            
        Returns:
            Database backend or None
        """
        db_name = name or self._default
        if db_name is None:
            return None
        return self._databases.get(db_name)
    
    def connect(self, name: str) -> bool:
        """Connect a specific database."""
        db = self._databases.get(name)
        return db.connect() if db else False
    
    def connect_all(self) -> Dict[str, bool]:
        """Connect all databases."""
        results = {}
        for name, db in self._databases.items():
            results[name] = db.connect()
        return results
    
    def disconnect_all(self) -> None:
        """Disconnect all databases."""
        for db in self._databases.values():
            db.disconnect()
    
    def list_databases(self) -> List[str]:
        """List registered database names."""
        return list(self._databases.keys())


# Global database manager
_global_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = DatabaseManager()
    return _global_manager


def get_database(name: Optional[str] = None) -> Optional[DatabaseBackend]:
    """Get database from global manager."""
    return get_database_manager().get(name)


def register_database(
    name: str,
    backend: DatabaseBackend,
    default: bool = False
) -> None:
    """Register database in global manager."""
    get_database_manager().register(name, backend, default)
