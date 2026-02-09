"""
Stream Integration for Enigma AI Engine

Integration with Twitch, YouTube, and Discord streaming.

Features:
- Chat reading
- Chat responses
- Stream overlays
- Viewer interaction
- Event handling

Usage:
    from enigma_engine.tools.stream_integration import StreamManager, get_stream_manager
    
    manager = get_stream_manager()
    
    # Connect to Twitch
    manager.connect_twitch(token="...", channel="mychannel")
    
    # Register chat handler
    manager.on_chat(lambda msg: print(f"{msg.user}: {msg.text}"))
"""

import asyncio
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Streaming platforms."""
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    DISCORD = "discord"


class MessageType(Enum):
    """Types of chat messages."""
    CHAT = "chat"
    COMMAND = "command"
    SUBSCRIPTION = "subscription"
    DONATION = "donation"
    RAID = "raid"
    HOST = "host"
    FOLLOW = "follow"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A chat message from stream."""
    platform: Platform
    user: str
    text: str
    message_type: MessageType = MessageType.CHAT
    
    # User info
    user_id: Optional[str] = None
    is_mod: bool = False
    is_sub: bool = False
    is_vip: bool = False
    
    # Message info
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None
    
    # Command parsing
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    
    # Extra data
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """A stream event."""
    platform: Platform
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StreamConfig:
    """Configuration for stream connection."""
    platform: Platform
    
    # Credentials
    token: str = ""
    channel: str = ""
    
    # Behavior
    auto_reconnect: bool = True
    command_prefix: str = "!"
    
    # Rate limiting
    message_cooldown: float = 1.0
    max_messages_per_minute: int = 20


class CommandHandler:
    """Handle chat commands."""
    
    def __init__(self, prefix: str = "!"):
        self._prefix = prefix
        self._commands: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        self._cooldowns: Dict[str, Dict[str, float]] = {}
    
    def register(
        self,
        name: str,
        handler: Callable[[ChatMessage], Optional[str]],
        aliases: Optional[List[str]] = None,
        cooldown: float = 0.0,
        mod_only: bool = False
    ):
        """
        Register a command.
        
        Args:
            name: Command name (without prefix)
            handler: Function that returns response string
            aliases: Alternative names
            cooldown: Per-user cooldown in seconds
            mod_only: Require moderator
        """
        self._commands[name] = {
            "handler": handler,
            "cooldown": cooldown,
            "mod_only": mod_only
        }
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
        
        self._cooldowns[name] = {}
    
    def parse_command(self, message: ChatMessage) -> bool:
        """
        Parse command from message.
        
        Returns:
            True if message is a command
        """
        if not message.text.startswith(self._prefix):
            return False
        
        parts = message.text[len(self._prefix):].split()
        if not parts:
            return False
        
        message.message_type = MessageType.COMMAND
        message.command = parts[0].lower()
        message.args = parts[1:]
        
        return True
    
    def execute(
        self,
        message: ChatMessage
    ) -> Optional[str]:
        """
        Execute a command.
        
        Returns:
            Response string or None
        """
        if not message.command:
            return None
        
        cmd_name = message.command
        
        # Check alias
        if cmd_name in self._aliases:
            cmd_name = self._aliases[cmd_name]
        
        if cmd_name not in self._commands:
            return None
        
        cmd = self._commands[cmd_name]
        
        # Check mod requirement
        if cmd["mod_only"] and not message.is_mod:
            return "This command requires moderator permissions."
        
        # Check cooldown
        if cmd["cooldown"] > 0:
            last_use = self._cooldowns[cmd_name].get(message.user, 0)
            if time.time() - last_use < cmd["cooldown"]:
                return None  # Silently ignore
            self._cooldowns[cmd_name][message.user] = time.time()
        
        try:
            return cmd["handler"](message)
        except Exception as e:
            logger.error(f"Command error: {e}")
            return f"Error executing command: {e}"


class PlatformConnection:
    """Base class for platform connections."""
    
    def __init__(self, config: StreamConfig):
        self._config = config
        self._connected = False
        self._message_queue: queue.Queue = queue.Queue()
    
    async def connect(self):
        """Connect to platform."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from platform."""
        raise NotImplementedError
    
    async def send_message(self, text: str):
        """Send a chat message."""
        raise NotImplementedError
    
    async def listen(self):
        """Listen for messages."""
        raise NotImplementedError
    
    def is_connected(self) -> bool:
        return self._connected


class TwitchConnection(PlatformConnection):
    """Connection to Twitch IRC."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self._reader = None
        self._writer = None
    
    async def connect(self):
        """Connect to Twitch IRC."""
        try:
            self._reader, self._writer = await asyncio.open_connection(
                'irc.chat.twitch.tv', 6667
            )
            
            # Authenticate
            self._writer.write(f"PASS oauth:{self._config.token}\r\n".encode())
            self._writer.write(f"NICK {self._config.channel}\r\n".encode())
            self._writer.write(f"JOIN #{self._config.channel}\r\n".encode())
            
            # Request capabilities
            self._writer.write("CAP REQ :twitch.tv/membership\r\n".encode())
            self._writer.write("CAP REQ :twitch.tv/tags\r\n".encode())
            self._writer.write("CAP REQ :twitch.tv/commands\r\n".encode())
            
            await self._writer.drain()
            
            self._connected = True
            logger.info(f"Connected to Twitch channel: {self._config.channel}")
            
        except Exception as e:
            logger.error(f"Twitch connection failed: {e}")
            self._connected = False
    
    async def disconnect(self):
        """Disconnect from Twitch."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
    
    async def send_message(self, text: str):
        """Send message to Twitch chat."""
        if not self._connected or not self._writer:
            return
        
        self._writer.write(f"PRIVMSG #{self._config.channel} :{text}\r\n".encode())
        await self._writer.drain()
    
    async def listen(self):
        """Listen for Twitch messages."""
        if not self._reader:
            return
        
        while self._connected:
            try:
                data = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=300  # 5 min timeout
                )
                
                if not data:
                    break
                
                line = data.decode('utf-8').strip()
                
                # Handle PING
                if line.startswith('PING'):
                    self._writer.write("PONG :tmi.twitch.tv\r\n".encode())
                    await self._writer.drain()
                    continue
                
                # Parse message
                message = self._parse_message(line)
                if message:
                    self._message_queue.put(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Twitch listen error: {e}")
                if self._config.auto_reconnect:
                    await self.connect()
    
    def _parse_message(self, line: str) -> Optional[ChatMessage]:
        """Parse Twitch IRC message."""
        # Basic PRIVMSG parsing
        match = re.match(
            r':(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)',
            line
        )
        
        if match:
            return ChatMessage(
                platform=Platform.TWITCH,
                user=match.group(1),
                text=match.group(2)
            )
        
        return None


class YouTubeConnection(PlatformConnection):
    """Connection to YouTube Live Chat."""
    
    async def connect(self):
        """Connect to YouTube API."""
        # Requires YouTube Data API
        logger.info("YouTube connection requires API setup")
        self._connected = True
    
    async def disconnect(self):
        self._connected = False
    
    async def send_message(self, text: str):
        """Send YouTube chat message."""
        # Would use YouTube Live Streaming API
        logger.debug(f"Would send to YouTube: {text}")
    
    async def listen(self):
        """Poll YouTube live chat."""
        while self._connected:
            await asyncio.sleep(5)  # Poll interval


class DiscordConnection(PlatformConnection):
    """Connection to Discord."""
    
    async def connect(self):
        """Connect to Discord."""
        # Requires discord.py
        logger.info("Discord connection requires discord.py")
        self._connected = True
    
    async def disconnect(self):
        self._connected = False
    
    async def send_message(self, text: str):
        """Send Discord message."""
        logger.debug(f"Would send to Discord: {text}")
    
    async def listen(self):
        """Listen for Discord messages."""
        while self._connected:
            await asyncio.sleep(1)


class StreamManager:
    """Manage stream integrations."""
    
    def __init__(self):
        self._connections: Dict[Platform, PlatformConnection] = {}
        self._commands = CommandHandler()
        
        # Callbacks
        self._chat_handlers: List[Callable[[ChatMessage], None]] = []
        self._event_handlers: List[Callable[[StreamEvent], None]] = []
        
        # State
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Rate limiting
        self._last_send: Dict[Platform, float] = {}
    
    def connect_twitch(
        self,
        token: str,
        channel: str,
        **kwargs
    ):
        """
        Connect to Twitch.
        
        Args:
            token: OAuth token
            channel: Channel name
        """
        config = StreamConfig(
            platform=Platform.TWITCH,
            token=token,
            channel=channel,
            **kwargs
        )
        
        conn = TwitchConnection(config)
        self._connections[Platform.TWITCH] = conn
        
        self._start_if_needed()
    
    def connect_youtube(
        self,
        token: str,
        channel: str,
        **kwargs
    ):
        """Connect to YouTube Live."""
        config = StreamConfig(
            platform=Platform.YOUTUBE,
            token=token,
            channel=channel,
            **kwargs
        )
        
        conn = YouTubeConnection(config)
        self._connections[Platform.YOUTUBE] = conn
        
        self._start_if_needed()
    
    def connect_discord(
        self,
        token: str,
        channel: str,
        **kwargs
    ):
        """Connect to Discord."""
        config = StreamConfig(
            platform=Platform.DISCORD,
            token=token,
            channel=channel,
            **kwargs
        )
        
        conn = DiscordConnection(config)
        self._connections[Platform.DISCORD] = conn
        
        self._start_if_needed()
    
    def disconnect(self, platform: Platform):
        """Disconnect from a platform."""
        if platform in self._connections:
            # Schedule disconnect
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._connections[platform].disconnect(),
                    self._loop
                )
            del self._connections[platform]
    
    def disconnect_all(self):
        """Disconnect from all platforms."""
        for platform in list(self._connections.keys()):
            self.disconnect(platform)
        
        self._running = False
    
    def on_chat(
        self,
        handler: Callable[[ChatMessage], None]
    ):
        """Register chat message handler."""
        self._chat_handlers.append(handler)
    
    def on_event(
        self,
        handler: Callable[[StreamEvent], None]
    ):
        """Register event handler."""
        self._event_handlers.append(handler)
    
    def register_command(
        self,
        name: str,
        handler: Callable[[ChatMessage], Optional[str]],
        **kwargs
    ):
        """Register a chat command."""
        self._commands.register(name, handler, **kwargs)
    
    def send_message(
        self,
        text: str,
        platform: Optional[Platform] = None
    ):
        """
        Send a message to chat.
        
        Args:
            text: Message text
            platform: Specific platform (or all)
        """
        targets = [platform] if platform else list(self._connections.keys())
        
        for plat in targets:
            if plat in self._connections and self._loop:
                # Rate limiting
                last = self._last_send.get(plat, 0)
                if time.time() - last < 1.0:  # 1 second minimum
                    continue
                
                self._last_send[plat] = time.time()
                
                asyncio.run_coroutine_threadsafe(
                    self._connections[plat].send_message(text),
                    self._loop
                )
    
    def _start_if_needed(self):
        """Start the async loop if not running."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def _run_loop(self):
        """Run the async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"Stream loop error: {e}")
        finally:
            self._loop.close()
    
    async def _main_loop(self):
        """Main async loop."""
        # Connect all
        for conn in self._connections.values():
            await conn.connect()
        
        # Start listeners
        tasks = [
            asyncio.create_task(conn.listen())
            for conn in self._connections.values()
        ]
        
        # Process messages
        while self._running:
            for conn in self._connections.values():
                while not conn._message_queue.empty():
                    try:
                        message = conn._message_queue.get_nowait()
                        self._process_message(message)
                    except queue.Empty:
                        break
            
            await asyncio.sleep(0.1)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
    
    def _process_message(self, message: ChatMessage):
        """Process an incoming message."""
        # Check for commands
        if self._commands.parse_command(message):
            response = self._commands.execute(message)
            if response:
                self.send_message(response, message.platform)
        
        # Call handlers
        for handler in self._chat_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Chat handler error: {e}")
    
    def get_connected_platforms(self) -> List[Platform]:
        """Get list of connected platforms."""
        return [
            p for p, c in self._connections.items()
            if c.is_connected()
        ]


# Global instance
_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get or create global stream manager."""
    global _manager
    if _manager is None:
        _manager = StreamManager()
    return _manager
