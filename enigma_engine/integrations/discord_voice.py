"""
Discord Voice Chat Integration
==============================

Real-time voice conversations with Enigma AI through Discord.
Supports both text and voice channels.

Features:
- Voice chat with AI (join voice channel, listen, respond)
- Text chat integration
- Slash commands
- Multi-server support
- Voice activity detection
- TTS responses

Requirements:
    pip install discord.py[voice] PyNaCl

Usage:
    from enigma_engine.integrations.discord_voice import DiscordBot
    
    bot = DiscordBot(token="YOUR_BOT_TOKEN")
    bot.run()

Or via CLI:
    python -m enigma_engine.integrations.discord_voice --token YOUR_TOKEN
"""

import logging
import asyncio
import io
import wave
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
import time
from collections import deque

logger = logging.getLogger(__name__)

# Check for discord.py
HAVE_DISCORD = False
HAVE_PYNACL = False

try:
    import discord
    from discord.ext import commands
    from discord import app_commands
    HAVE_DISCORD = True
except ImportError:
    discord = None
    commands = None
    app_commands = None

try:
    HAVE_PYNACL = True
except ImportError:
    pass  # Intentionally silent


class VoiceState(Enum):
    """Voice connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LISTENING = "listening"
    SPEAKING = "speaking"


@dataclass
class VoiceConfig:
    """Voice chat configuration."""
    # Voice settings
    sample_rate: int = 48000  # Discord uses 48kHz
    channels: int = 2  # Stereo
    frame_length: int = 20  # ms per frame
    
    # Silence detection
    silence_threshold: float = 0.01  # RMS threshold for silence
    silence_duration: float = 1.5  # Seconds of silence before processing
    max_recording_duration: float = 30.0  # Max seconds to record
    
    # TTS settings
    tts_enabled: bool = True
    tts_voice: str = "default"
    tts_speed: float = 1.0
    
    # Behavior
    auto_join: bool = False  # Auto-join when mentioned
    respond_to_text: bool = True  # Respond to text messages too
    prefix: str = "!"  # Command prefix


@dataclass
class ConversationContext:
    """Track conversation context per channel/user."""
    user_id: int
    channel_id: int
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_interaction: float = field(default_factory=time.time)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to context."""
        self.messages.append({"role": role, "content": content})
        # Keep last 20 messages
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
        self.last_interaction = time.time()


class AudioBuffer:
    """Buffer for collecting audio from voice channel."""
    
    def __init__(self, config: VoiceConfig):
        self.config = config
        self.buffer: deque = deque()
        self.is_recording = False
        self.silence_start: Optional[float] = None
        self.recording_start: Optional[float] = None
    
    def add_pcm(self, pcm_data: bytes, user_id: int) -> None:
        """Add PCM audio data to buffer."""
        if not self.is_recording:
            return
        
        self.buffer.append({
            'data': pcm_data,
            'user_id': user_id,
            'timestamp': time.time(),
        })
    
    def start_recording(self) -> None:
        """Start recording."""
        self.buffer.clear()
        self.is_recording = True
        self.silence_start = None
        self.recording_start = time.time()
    
    def stop_recording(self) -> bytes:
        """Stop recording and return combined audio."""
        self.is_recording = False
        
        # Combine all audio chunks
        all_data = b''.join(chunk['data'] for chunk in self.buffer)
        self.buffer.clear()
        
        return all_data
    
    def get_duration(self) -> float:
        """Get current recording duration."""
        if self.recording_start is None:
            return 0.0
        return time.time() - self.recording_start
    
    def is_silence(self, pcm_data: bytes) -> bool:
        """Check if audio chunk is silence."""
        # Calculate RMS
        samples = [int.from_bytes(pcm_data[i:i+2], 'little', signed=True) 
                   for i in range(0, len(pcm_data), 2)]
        if not samples:
            return True
        
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
        normalized_rms = rms / 32768.0
        
        return normalized_rms < self.config.silence_threshold


if HAVE_DISCORD:
    class VoiceSink(discord.sinks.WaveSink):
        """Custom sink for receiving voice audio."""
        
        def __init__(self, buffer: AudioBuffer, target_user: Optional[int] = None):
            super().__init__()
            self.buffer = buffer
            self.target_user = target_user
        
        def write(self, data: bytes, user: int) -> None:
            """Receive audio data from a user."""
            # Only record from target user if specified
            if self.target_user is not None and user != self.target_user:
                return
            
            self.buffer.add_pcm(data, user)
    
    
    class EnigmaVoiceClient:
        """Voice client for Enigma AI in Discord."""
        
        def __init__(
            self,
            voice_client: discord.VoiceClient,
            config: VoiceConfig,
            on_audio_complete: Callable[[bytes], None],
        ):
            self.voice_client = voice_client
            self.config = config
            self.on_audio_complete = on_audio_complete
            
            self.state = VoiceState.CONNECTED
            self.buffer = AudioBuffer(config)
            self._listening_task: Optional[asyncio.Task] = None
        
        async def start_listening(self, user_id: Optional[int] = None) -> None:
            """Start listening for voice input."""
            if self.state == VoiceState.LISTENING:
                return
            
            self.state = VoiceState.LISTENING
            self.buffer.start_recording()
            
            # Create sink
            sink = VoiceSink(self.buffer, target_user=user_id)
            
            # Start recording
            self.voice_client.start_recording(
                sink,
                self._on_recording_complete,
                user_id,
            )
            
            logger.info(f"Started listening in voice channel")
        
        def _on_recording_complete(self, sink: discord.sinks.Sink, *args):
            """Called when recording is complete."""
            audio_data = self.buffer.stop_recording()
            self.state = VoiceState.CONNECTED
            
            if audio_data and len(audio_data) > 1000:  # Minimum audio length
                self.on_audio_complete(audio_data)
        
        async def stop_listening(self) -> bytes:
            """Stop listening and return audio."""
            if self.state != VoiceState.LISTENING:
                return b''
            
            self.voice_client.stop_recording()
            return self.buffer.stop_recording()
        
        async def speak(self, audio_source: discord.AudioSource) -> None:
            """Play audio in voice channel."""
            if self.voice_client.is_playing():
                self.voice_client.stop()
            
            self.state = VoiceState.SPEAKING
            self.voice_client.play(
                audio_source,
                after=lambda e: self._on_speak_complete(e)
            )
        
        def _on_speak_complete(self, error: Optional[Exception]) -> None:
            """Called when speaking is complete."""
            if error:
                logger.error(f"Error during speech: {error}")
            self.state = VoiceState.CONNECTED
        
        async def disconnect(self) -> None:
            """Disconnect from voice channel."""
            if self.voice_client.is_connected():
                await self.voice_client.disconnect()
            self.state = VoiceState.DISCONNECTED


class DiscordBot:
    """
    Discord bot with voice chat integration for Enigma AI.
    
    Example:
        bot = DiscordBot(token="YOUR_TOKEN")
        bot.set_inference_callback(my_inference_fn)
        bot.run()
    """
    
    def __init__(
        self,
        token: str,
        config: Optional[VoiceConfig] = None,
        inference_callback: Optional[Callable[[str, List[Dict]], str]] = None,
    ):
        """
        Initialize Discord bot.
        
        Args:
            token: Discord bot token
            config: Voice configuration
            inference_callback: Function(prompt, history) -> response
        """
        if not HAVE_DISCORD:
            raise RuntimeError(
                "Discord voice requires discord.py[voice]. "
                "Install with: pip install discord.py[voice] PyNaCl"
            )
        
        self.token = token
        self.config = config or VoiceConfig()
        self.inference_callback = inference_callback
        
        # Conversation contexts per channel
        self._contexts: Dict[int, ConversationContext] = {}
        
        # Voice clients per guild
        self._voice_clients: Dict[int, EnigmaVoiceClient] = {}
        
        # Create bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        
        self.bot = commands.Bot(
            command_prefix=self.config.prefix,
            intents=intents,
        )
        
        # Setup handlers
        self._setup_events()
        self._setup_commands()
    
    def set_inference_callback(
        self,
        callback: Callable[[str, List[Dict]], str],
    ) -> None:
        """Set the callback function for AI inference."""
        self.inference_callback = callback
    
    def _get_context(self, user_id: int, channel_id: int) -> ConversationContext:
        """Get or create conversation context."""
        if channel_id not in self._contexts:
            self._contexts[channel_id] = ConversationContext(
                user_id=user_id,
                channel_id=channel_id,
            )
        return self._contexts[channel_id]
    
    async def _generate_response(
        self,
        prompt: str,
        context: ConversationContext,
    ) -> str:
        """Generate AI response."""
        if self.inference_callback:
            # Add user message to context
            context.add_message("user", prompt)
            
            # Get response
            response = self.inference_callback(prompt, context.messages)
            
            # Add assistant response to context
            context.add_message("assistant", response)
            
            return response
        else:
            return "I'm not configured for inference yet. Please set up the inference callback."
    
    def _setup_events(self) -> None:
        """Setup event handlers."""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Logged in as {self.bot.user}")
            
            # Sync slash commands
            try:
                synced = await self.bot.tree.sync()
                logger.info(f"Synced {len(synced)} command(s)")
            except Exception as e:
                logger.error(f"Failed to sync commands: {e}")
        
        @self.bot.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self.bot.user:
                return
            
            # Process commands first
            await self.bot.process_commands(message)
            
            # Respond to mentions
            if self.bot.user in message.mentions:
                # Remove mention from content
                content = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
                
                if content:
                    context = self._get_context(message.author.id, message.channel.id)
                    
                    async with message.channel.typing():
                        response = await self._generate_response(content, context)
                    
                    # Split long responses
                    if len(response) > 2000:
                        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(response)
        
        @self.bot.event
        async def on_voice_state_update(
            member: discord.Member,
            before: discord.VoiceState,
            after: discord.VoiceState,
        ):
            # Handle bot disconnection
            if member == self.bot.user and after.channel is None:
                if before.channel and before.channel.guild.id in self._voice_clients:
                    del self._voice_clients[before.channel.guild.id]
                    logger.info(f"Disconnected from voice in {before.channel.guild}")
    
    def _setup_commands(self) -> None:
        """Setup bot commands."""
        
        # Slash commands
        @self.bot.tree.command(name="chat", description="Chat with Enigma AI")
        @app_commands.describe(message="Your message to the AI")
        async def chat(interaction: discord.Interaction, message: str):
            context = self._get_context(interaction.user.id, interaction.channel_id)
            
            await interaction.response.defer()
            response = await self._generate_response(message, context)
            
            if len(response) > 2000:
                response = response[:1997] + "..."
            
            await interaction.followup.send(response)
        
        @self.bot.tree.command(name="join", description="Join your voice channel")
        async def join(interaction: discord.Interaction):
            if not interaction.user.voice:
                await interaction.response.send_message(
                    "You need to be in a voice channel!",
                    ephemeral=True
                )
                return
            
            channel = interaction.user.voice.channel
            
            # Connect to voice
            try:
                voice_client = await channel.connect()
                
                # Create voice handler
                self._voice_clients[interaction.guild_id] = EnigmaVoiceClient(
                    voice_client,
                    self.config,
                    lambda audio: asyncio.create_task(
                        self._handle_voice_input(interaction.guild_id, audio)
                    ),
                )
                
                await interaction.response.send_message(
                    f"Joined {channel.name}! Say 'Hey Enigma' to talk to me.",
                    ephemeral=True
                )
                logger.info(f"Joined voice channel: {channel.name}")
                
            except Exception as e:
                await interaction.response.send_message(
                    f"Failed to join: {e}",
                    ephemeral=True
                )
        
        @self.bot.tree.command(name="leave", description="Leave voice channel")
        async def leave(interaction: discord.Interaction):
            if interaction.guild_id in self._voice_clients:
                await self._voice_clients[interaction.guild_id].disconnect()
                del self._voice_clients[interaction.guild_id]
                
                await interaction.response.send_message(
                    "Left voice channel!",
                    ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "I'm not in a voice channel!",
                    ephemeral=True
                )
        
        @self.bot.tree.command(name="listen", description="Start listening in voice")
        async def listen(interaction: discord.Interaction):
            if interaction.guild_id not in self._voice_clients:
                await interaction.response.send_message(
                    "I need to join a voice channel first! Use /join",
                    ephemeral=True
                )
                return
            
            client = self._voice_clients[interaction.guild_id]
            await client.start_listening(interaction.user.id)
            
            await interaction.response.send_message(
                "Listening... Speak your message!",
                ephemeral=True
            )
            
            # Auto-stop after max duration
            await asyncio.sleep(self.config.max_recording_duration)
            if client.state == VoiceState.LISTENING:
                audio = await client.stop_listening()
                if audio:
                    await self._handle_voice_input(interaction.guild_id, audio)
        
        @self.bot.tree.command(name="stop", description="Stop listening")
        async def stop(interaction: discord.Interaction):
            if interaction.guild_id in self._voice_clients:
                client = self._voice_clients[interaction.guild_id]
                audio = await client.stop_listening()
                
                await interaction.response.send_message(
                    "Stopped listening!",
                    ephemeral=True
                )
                
                if audio:
                    await self._handle_voice_input(interaction.guild_id, audio)
            else:
                await interaction.response.send_message(
                    "Not currently listening!",
                    ephemeral=True
                )
        
        @self.bot.tree.command(name="clear", description="Clear conversation history")
        async def clear(interaction: discord.Interaction):
            if interaction.channel_id in self._contexts:
                self._contexts[interaction.channel_id].messages.clear()
            
            await interaction.response.send_message(
                "Conversation cleared!",
                ephemeral=True
            )
        
        # Prefix commands
        @self.bot.command(name="ask")
        async def ask(ctx: commands.Context, *, question: str):
            """Ask the AI a question."""
            context = self._get_context(ctx.author.id, ctx.channel.id)
            
            async with ctx.typing():
                response = await self._generate_response(question, context)
            
            await ctx.reply(response)
    
    async def _handle_voice_input(self, guild_id: int, audio_data: bytes) -> None:
        """Process voice input and generate response."""
        try:
            # Get voice client
            client = self._voice_clients.get(guild_id)
            if not client:
                return
            
            # Convert audio to text (STT)
            text = await self._speech_to_text(audio_data)
            
            if not text or len(text.strip()) < 2:
                return
            
            logger.info(f"Voice input: {text}")
            
            # Get text channel for context
            voice_channel = client.voice_client.channel
            text_channel = voice_channel.guild.system_channel or voice_channel.guild.text_channels[0]
            
            # Generate response
            context = self._get_context(0, text_channel.id)  # 0 = voice user
            response = await self._generate_response(text, context)
            
            # Convert to speech (TTS)
            if self.config.tts_enabled:
                audio_response = await self._text_to_speech(response)
                
                if audio_response:
                    # Play response
                    audio_source = discord.FFmpegPCMAudio(
                        io.BytesIO(audio_response),
                        pipe=True,
                    )
                    await client.speak(audio_source)
            
            # Also send to text channel
            if len(response) <= 2000:
                await text_channel.send(f"**Voice response:** {response}")
            
        except Exception as e:
            logger.error(f"Error handling voice input: {e}")
    
    async def _speech_to_text(self, audio_data: bytes) -> str:
        """Convert audio to text using STT."""
        try:
            # Try to use the Enigma voice listener
            from ..voice.listener import VoiceListener
            
            listener = VoiceListener()
            
            # Save audio to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                # Write WAV header
                with wave.open(f, 'wb') as wav:
                    wav.setnchannels(self.config.channels)
                    wav.setsampwidth(2)  # 16-bit
                    wav.setframerate(self.config.sample_rate)
                    wav.writeframes(audio_data)
                
                temp_path = f.name
            
            # Transcribe
            text = listener.transcribe_file(temp_path)
            
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
            
            return text or ""
            
        except ImportError:
            # Fallback: try whisper directly
            try:
                import whisper
                
                model = whisper.load_model("base")
                
                # Save temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    with wave.open(f, 'wb') as wav:
                        wav.setnchannels(self.config.channels)
                        wav.setsampwidth(2)
                        wav.setframerate(self.config.sample_rate)
                        wav.writeframes(audio_data)
                    temp_path = f.name
                
                result = model.transcribe(temp_path)
                Path(temp_path).unlink(missing_ok=True)
                
                return result.get('text', '')
                
            except ImportError:
                logger.warning("No STT available. Install whisper: pip install openai-whisper")
                return ""
    
    async def _text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio."""
        try:
            # Try Enigma voice generator
            from ..voice.voice_generator import AIVoiceGenerator
            
            generator = AIVoiceGenerator()
            audio = generator.generate(text)
            
            if audio:
                return audio
            
        except ImportError:
            pass  # Intentionally silent
        
        try:
            # Fallback: pyttsx3
            import pyttsx3
            import tempfile
            
            engine = pyttsx3.init()
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            with open(temp_path, 'rb') as f:
                audio = f.read()
            
            Path(temp_path).unlink(missing_ok=True)
            return audio
            
        except ImportError:
            logger.warning("No TTS available. Install pyttsx3: pip install pyttsx3")
            return None
    
    def run(self) -> None:
        """Run the bot (blocking)."""
        logger.info("Starting Discord bot...")
        self.bot.run(self.token)
    
    async def start(self) -> None:
        """Start the bot (async)."""
        await self.bot.start(self.token)
    
    async def close(self) -> None:
        """Close the bot."""
        # Disconnect all voice clients
        for client in self._voice_clients.values():
            await client.disconnect()
        
        await self.bot.close()


def create_bot(
    token: str,
    inference_fn: Optional[Callable[[str, List[Dict]], str]] = None,
    **config_kwargs,
) -> DiscordBot:
    """
    Create a Discord bot instance.
    
    Args:
        token: Discord bot token
        inference_fn: Function(prompt, history) -> response
        **config_kwargs: VoiceConfig parameters
    
    Returns:
        Configured DiscordBot instance
    """
    config = VoiceConfig(**config_kwargs)
    bot = DiscordBot(token=token, config=config, inference_callback=inference_fn)
    return bot


def run_bot(token: str, inference_fn: Optional[Callable] = None) -> None:
    """Convenience function to run the bot."""
    bot = create_bot(token, inference_fn)
    bot.run()


# Integration with Enigma inference
def create_enigma_bot(token: str) -> DiscordBot:
    """
    Create a Discord bot integrated with Enigma AI.
    
    Automatically uses the Enigma inference engine for responses.
    """
    try:
        from ..core.inference import EnigmaEngine
        
        # Create engine
        engine = EnigmaEngine()
        
        def inference_callback(prompt: str, history: List[Dict]) -> str:
            # Format history for context
            context = "\n".join(
                f"{msg['role'].title()}: {msg['content']}"
                for msg in history[-10:]  # Last 10 messages
            )
            
            full_prompt = f"{context}\nUser: {prompt}\nAssistant:"
            return engine.generate(full_prompt)
        
        return create_bot(token, inference_callback)
        
    except ImportError:
        logger.warning("Enigma engine not available, using basic bot")
        return create_bot(token)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Enigma Discord Bot")
    parser.add_argument("--token", required=True, help="Discord bot token")
    parser.add_argument("--prefix", default="!", help="Command prefix")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS")
    
    args = parser.parse_args()
    
    config = VoiceConfig(
        prefix=args.prefix,
        tts_enabled=not args.no_tts,
    )
    
    bot = create_bot(args.token, tts_enabled=config.tts_enabled, prefix=config.prefix)
    bot.run()


# Export public API
__all__ = [
    'DiscordBot',
    'VoiceConfig',
    'VoiceState',
    'ConversationContext',
    'create_bot',
    'run_bot',
    'create_enigma_bot',
    'HAVE_DISCORD',
    'HAVE_PYNACL',
]
