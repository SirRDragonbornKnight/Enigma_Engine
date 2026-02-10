"""
Stream Integration

Integration with Twitch and YouTube streaming platforms.
Supports chat reading, commands, alerts, and AI-powered chat interaction.

FILE: enigma_engine/game/streaming.py
TYPE: Game
MAIN CLASSES: TwitchIntegration, YouTubeIntegration, StreamManager
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import Any, Callable, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamPlatform(Enum):
    """Streaming platforms."""
    TWITCH = "twitch"
    YOUTUBE = "youtube"


class MessageType(Enum):
    """Types of chat messages."""
    CHAT = "chat"
    COMMAND = "command"
    SUBSCRIPTION = "subscription"
    DONATION = "donation"
    RAID = "raid"
    FOLLOW = "follow"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """A chat message from stream."""
    platform: StreamPlatform
    message_id: str
    user_id: str
    username: str
    display_name: str
    content: str
    message_type: MessageType = MessageType.CHAT
    timestamp: float = field(default_factory=time.time)
    is_mod: bool = False
    is_subscriber: bool = False
    is_vip: bool = False
    badges: list[str] = field(default_factory=list)
    emotes: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Configuration for stream integration."""
    # Twitch
    twitch_channel: str = ""
    twitch_oauth_token: str = ""
    twitch_client_id: str = ""
    
    # YouTube
    youtube_api_key: str = ""
    youtube_channel_id: str = ""
    youtube_live_chat_id: str = ""
    
    # Bot settings
    command_prefix: str = "!"
    cooldown_seconds: float = 3.0
    
    # AI settings
    ai_respond_to_mentions: bool = True
    ai_respond_to_commands: bool = True
    ai_response_chance: float = 0.1  # Random response chance
    
    # Moderation
    blocked_words: list[str] = field(default_factory=list)
    allowed_users: set[str] = field(default_factory=set)
    mod_only_commands: set[str] = field(default_factory=set)


@dataclass
class ChatCommand:
    """A registered chat command."""
    name: str
    handler: Callable[[ChatMessage], Optional[str]]
    description: str = ""
    cooldown: float = 0
    mod_only: bool = False
    sub_only: bool = False
    last_used: float = 0


class TwitchIntegration:
    """
    Integration with Twitch chat using IRC.
    """
    
    IRC_SERVER = "irc.chat.twitch.tv"
    IRC_PORT = 6667
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._connected = False
        self._reader = None
        self._writer = None
        self._message_callback: Optional[Callable[[ChatMessage], None]] = None
        self._running = False
    
    async def connect(self):
        """Connect to Twitch IRC."""
        if not self.config.twitch_oauth_token or not self.config.twitch_channel:
            raise ValueError("Twitch OAuth token and channel required")
        
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self.IRC_SERVER, self.IRC_PORT
            )
            
            # Authenticate
            token = self.config.twitch_oauth_token
            if not token.startswith("oauth:"):
                token = f"oauth:{token}"
            
            self._writer.write(f"PASS {token}\r\n".encode())
            self._writer.write(f"NICK {self.config.twitch_channel}\r\n".encode())
            await self._writer.drain()
            
            # Request capabilities
            self._writer.write(b"CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership\r\n")
            await self._writer.drain()
            
            # Join channel
            channel = self.config.twitch_channel.lower()
            if not channel.startswith("#"):
                channel = f"#{channel}"
            
            self._writer.write(f"JOIN {channel}\r\n".encode())
            await self._writer.drain()
            
            self._connected = True
            logger.info(f"Connected to Twitch channel: {channel}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Twitch: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Twitch."""
        self._running = False
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
    
    async def read_messages(self):
        """Read and process messages from Twitch chat."""
        self._running = True
        
        while self._running and self._connected:
            try:
                line = await self._reader.readline()
                if not line:
                    break
                
                message = line.decode().strip()
                
                # Handle PING
                if message.startswith("PING"):
                    self._writer.write(b"PONG :tmi.twitch.tv\r\n")
                    await self._writer.drain()
                    continue
                
                # Parse PRIVMSG
                chat_msg = self._parse_message(message)
                if chat_msg and self._message_callback:
                    self._message_callback(chat_msg)
                    
            except Exception as e:
                logger.error(f"Error reading Twitch message: {e}")
                await asyncio.sleep(1)
    
    def _parse_message(self, raw: str) -> Optional[ChatMessage]:
        """Parse IRC message into ChatMessage."""
        # Format: @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
        
        if "PRIVMSG" not in raw:
            return None
        
        try:
            tags = {}
            if raw.startswith("@"):
                tag_end = raw.index(" ")
                tag_str = raw[1:tag_end]
                raw = raw[tag_end + 1:]
                
                for tag in tag_str.split(";"):
                    if "=" in tag:
                        key, value = tag.split("=", 1)
                        tags[key] = value
            
            # Parse user and message
            match = re.match(r":(\w+)!.+ PRIVMSG #\w+ :(.+)", raw)
            if not match:
                return None
            
            username = match.group(1)
            content = match.group(2)
            
            return ChatMessage(
                platform=StreamPlatform.TWITCH,
                message_id=tags.get("id", str(time.time())),
                user_id=tags.get("user-id", username),
                username=username.lower(),
                display_name=tags.get("display-name", username),
                content=content,
                is_mod=tags.get("mod") == "1",
                is_subscriber=tags.get("subscriber") == "1",
                is_vip=tags.get("vip") == "1",
                badges=[b.split("/")[0] for b in tags.get("badges", "").split(",") if b],
                emotes=tags.get("emotes", "").split("/") if tags.get("emotes") else [],
                raw_data=tags
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")
            return None
    
    async def send_message(self, content: str):
        """Send a message to Twitch chat."""
        if not self._connected:
            return
        
        channel = self.config.twitch_channel.lower()
        if not channel.startswith("#"):
            channel = f"#{channel}"
        
        self._writer.write(f"PRIVMSG {channel} :{content}\r\n".encode())
        await self._writer.drain()
    
    def set_message_callback(self, callback: Callable[[ChatMessage], None]):
        """Set callback for incoming messages."""
        self._message_callback = callback


class YouTubeIntegration:
    """
    Integration with YouTube Live Chat API.
    """
    
    API_BASE = "https://www.googleapis.com/youtube/v3"
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._message_callback: Optional[Callable[[ChatMessage], None]] = None
        self._running = False
        self._next_page_token: Optional[str] = None
        self._poll_interval: float = 5.0
    
    async def connect(self):
        """Initialize YouTube chat polling."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for YouTube integration")
        
        if not self.config.youtube_api_key:
            raise ValueError("YouTube API key required")
        
        # Get live chat ID if not provided
        if not self.config.youtube_live_chat_id:
            await self._get_live_chat_id()
        
        logger.info("YouTube Live Chat integration initialized")
    
    async def _get_live_chat_id(self):
        """Get the live chat ID for the channel's current stream."""
        async with aiohttp.ClientSession() as session:
            # Get live broadcast
            url = f"{self.API_BASE}/liveBroadcasts"
            params = {
                "part": "snippet",
                "broadcastStatus": "active",
                "broadcastType": "all",
                "key": self.config.youtube_api_key
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise ValueError(f"Failed to get broadcasts: {await resp.text()}")
                
                data = await resp.json()
                
                if data.get("items"):
                    # Get liveChatId from first active broadcast
                    broadcast = data["items"][0]
                    live_chat_id = broadcast["snippet"].get("liveChatId")
                    if live_chat_id:
                        self.config.youtube_live_chat_id = live_chat_id
                        return
                
                raise ValueError("No active live stream found")
    
    async def read_messages(self):
        """Poll for new chat messages."""
        if not HAS_AIOHTTP:
            return
        
        self._running = True
        
        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    url = f"{self.API_BASE}/liveChat/messages"
                    params = {
                        "liveChatId": self.config.youtube_live_chat_id,
                        "part": "snippet,authorDetails",
                        "key": self.config.youtube_api_key
                    }
                    
                    if self._next_page_token:
                        params["pageToken"] = self._next_page_token
                    
                    async with session.get(url, params=params) as resp:
                        if resp.status != 200:
                            logger.error(f"YouTube API error: {await resp.text()}")
                            await asyncio.sleep(10)
                            continue
                        
                        data = await resp.json()
                        
                        self._next_page_token = data.get("nextPageToken")
                        self._poll_interval = data.get("pollingIntervalMillis", 5000) / 1000
                        
                        for item in data.get("items", []):
                            msg = self._parse_message(item)
                            if msg and self._message_callback:
                                self._message_callback(msg)
                    
                    await asyncio.sleep(self._poll_interval)
                    
                except Exception as e:
                    logger.error(f"Error polling YouTube chat: {e}")
                    await asyncio.sleep(10)
    
    def _parse_message(self, item: dict[str, Any]) -> Optional[ChatMessage]:
        """Parse YouTube API message into ChatMessage."""
        try:
            snippet = item["snippet"]
            author = item["authorDetails"]
            
            return ChatMessage(
                platform=StreamPlatform.YOUTUBE,
                message_id=item["id"],
                user_id=author["channelId"],
                username=author["channelId"],
                display_name=author["displayName"],
                content=snippet.get("displayMessage", ""),
                message_type=self._get_message_type(snippet.get("type")),
                is_mod=author.get("isChatModerator", False),
                is_subscriber=author.get("isChatSponsor", False),
                badges=["owner"] if author.get("isChatOwner") else [],
                raw_data=item
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse YouTube message: {e}")
            return None
    
    def _get_message_type(self, yt_type: str) -> MessageType:
        """Convert YouTube message type to MessageType."""
        type_map = {
            "textMessageEvent": MessageType.CHAT,
            "superChatEvent": MessageType.DONATION,
            "superStickerEvent": MessageType.DONATION,
            "memberMilestoneChatEvent": MessageType.SUBSCRIPTION,
            "newSponsorEvent": MessageType.SUBSCRIPTION,
        }
        return type_map.get(yt_type, MessageType.CHAT)
    
    async def send_message(self, content: str):
        """Send a message to YouTube chat."""
        if not HAS_AIOHTTP:
            return
        
        # Note: Requires OAuth for sending messages
        # Using API key only allows reading
        logger.warning("YouTube send_message requires OAuth authentication")
    
    def set_message_callback(self, callback: Callable[[ChatMessage], None]):
        """Set callback for incoming messages."""
        self._message_callback = callback
    
    async def disconnect(self):
        """Stop polling."""
        self._running = False


class StreamManager:
    """
    Unified manager for stream integrations.
    Handles chat, commands, and AI responses.
    """
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        
        self._twitch: Optional[TwitchIntegration] = None
        self._youtube: Optional[YouTubeIntegration] = None
        
        self._commands: dict[str, ChatCommand] = {}
        self._message_handlers: list[Callable[[ChatMessage], None]] = []
        self._ai_responder: Optional[Callable[[str], str]] = None
        
        self._message_queue: Queue = Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
        # Default commands
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register default chat commands."""
        self.register_command(
            "help",
            lambda msg: "Available commands: " + ", ".join(f"!{c}" for c in self._commands.keys()),
            "Show available commands"
        )
        
        self.register_command(
            "ai",
            lambda msg: self._handle_ai_command(msg),
            "Ask the AI assistant"
        )
    
    def register_command(
        self,
        name: str,
        handler: Callable[[ChatMessage], Optional[str]],
        description: str = "",
        cooldown: float = 0,
        mod_only: bool = False,
        sub_only: bool = False
    ):
        """Register a chat command."""
        self._commands[name.lower()] = ChatCommand(
            name=name.lower(),
            handler=handler,
            description=description,
            cooldown=cooldown,
            mod_only=mod_only,
            sub_only=sub_only
        )
    
    def set_ai_responder(self, responder: Callable[[str], str]):
        """Set AI response generator."""
        self._ai_responder = responder
    
    def add_message_handler(self, handler: Callable[[ChatMessage], None]):
        """Add handler for all messages."""
        self._message_handlers.append(handler)
    
    def _handle_ai_command(self, msg: ChatMessage) -> Optional[str]:
        """Handle AI command."""
        if not self._ai_responder:
            return "AI not configured"
        
        # Remove command prefix
        question = msg.content
        if question.lower().startswith("!ai "):
            question = question[4:].strip()
        
        if not question:
            return "Please include a question after !ai"
        
        try:
            response = self._ai_responder(question)
            return f"@{msg.display_name} {response[:400]}"
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return "Sorry, I couldn't process that request"
    
    def _process_message(self, msg: ChatMessage):
        """Process incoming chat message."""
        # Call handlers
        for handler in self._message_handlers:
            try:
                handler(msg)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        
        # Check for commands
        if msg.content.startswith(self.config.command_prefix):
            self._handle_command(msg)
        
        # AI mention response
        elif self.config.ai_respond_to_mentions and self._ai_responder:
            # Check if bot name is mentioned
            bot_name = self.config.twitch_channel.lower()
            if bot_name in msg.content.lower():
                response = self._ai_responder(msg.content)
                if response:
                    self._send_message(msg.platform, f"@{msg.display_name} {response[:400]}")
    
    def _handle_command(self, msg: ChatMessage):
        """Handle a chat command."""
        parts = msg.content[len(self.config.command_prefix):].split(maxsplit=1)
        if not parts:
            return
        
        cmd_name = parts[0].lower()
        cmd = self._commands.get(cmd_name)
        
        if not cmd:
            return
        
        # Check permissions
        if cmd.mod_only and not msg.is_mod:
            return
        
        if cmd.sub_only and not msg.is_subscriber:
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - cmd.last_used < cmd.cooldown:
            return
        
        cmd.last_used = current_time
        
        # Execute command
        try:
            response = cmd.handler(msg)
            if response:
                self._send_message(msg.platform, response)
        except Exception as e:
            logger.error(f"Command error ({cmd_name}): {e}")
    
    def _send_message(self, platform: StreamPlatform, content: str):
        """Queue message for sending."""
        if self._loop:
            if platform == StreamPlatform.TWITCH and self._twitch:
                asyncio.run_coroutine_threadsafe(
                    self._twitch.send_message(content),
                    self._loop
                )
            elif platform == StreamPlatform.YOUTUBE and self._youtube:
                asyncio.run_coroutine_threadsafe(
                    self._youtube.send_message(content),
                    self._loop
                )
    
    def start(self):
        """Start stream manager in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def _run_loop(self):
        """Run async event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._async_main())
        finally:
            self._loop.close()
    
    async def _async_main(self):
        """Main async routine."""
        tasks = []
        
        # Connect to Twitch
        if self.config.twitch_channel and self.config.twitch_oauth_token:
            self._twitch = TwitchIntegration(self.config)
            self._twitch.set_message_callback(self._process_message)
            
            try:
                await self._twitch.connect()
                tasks.append(asyncio.create_task(self._twitch.read_messages()))
                logger.info("Twitch integration started")
            except Exception as e:
                logger.error(f"Twitch connection failed: {e}")
        
        # Connect to YouTube
        if self.config.youtube_api_key:
            self._youtube = YouTubeIntegration(self.config)
            self._youtube.set_message_callback(self._process_message)
            
            try:
                await self._youtube.connect()
                tasks.append(asyncio.create_task(self._youtube.read_messages()))
                logger.info("YouTube integration started")
            except Exception as e:
                logger.error(f"YouTube connection failed: {e}")
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks)
    
    def stop(self):
        """Stop stream manager."""
        self._running = False
        
        if self._loop:
            if self._twitch:
                asyncio.run_coroutine_threadsafe(
                    self._twitch.disconnect(),
                    self._loop
                )
            if self._youtube:
                asyncio.run_coroutine_threadsafe(
                    self._youtube.disconnect(),
                    self._loop
                )
        
        if self._thread:
            self._thread.join(timeout=5)


def create_stream_manager(
    twitch_channel: str = None,
    twitch_token: str = None,
    youtube_api_key: str = None,
    ai_responder: Callable[[str], str] = None
) -> StreamManager:
    """
    Create and configure a stream manager.
    
    Args:
        twitch_channel: Twitch channel name
        twitch_token: Twitch OAuth token
        youtube_api_key: YouTube Data API key
        ai_responder: Function for AI responses
    
    Returns:
        Configured StreamManager
    """
    config = StreamConfig(
        twitch_channel=twitch_channel or "",
        twitch_oauth_token=twitch_token or "",
        youtube_api_key=youtube_api_key or ""
    )
    
    manager = StreamManager(config)
    
    if ai_responder:
        manager.set_ai_responder(ai_responder)
    
    return manager
