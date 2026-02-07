"""
Discord Bot Template for Enigma AI Engine

A ready-to-deploy Discord bot that integrates with Enigma AI Engine.

Features:
- Chat with AI in channels or DMs
- Slash commands for various AI tasks
- Image generation
- Code generation
- Voice chat integration (coming soon)

Usage:
    1. Create a Discord application at https://discord.com/developers
    2. Create a bot and get the token
    3. Set DISCORD_TOKEN and FORGE_API_URL environment variables
    4. Run: python discord_bot.py
"""

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Optional imports
try:
    import discord
    from discord import app_commands
    from discord.ext import commands
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Bot configuration"""
    token: str
    forge_api_url: str = "http://localhost:8000"
    forge_api_key: str = ""
    command_prefix: str = "!"
    max_response_length: int = 2000
    allowed_channels: List[int] = None  # None = all channels
    admin_users: List[int] = None
    rate_limit_per_user: int = 10  # requests per minute


class Enigma AI EngineClient:
    """Client for communicating with Enigma AI Engine API"""
    
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def chat(self, message: str, 
                  conversation_id: str = None,
                  system_prompt: str = None) -> str:
        """Send a chat message and get response"""
        session = await self.get_session()
        
        payload = {
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with session.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    error = await response.text()
                    logger.error(f"Chat API error: {error}")
                    return f"Error: API returned {response.status}"
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    async def generate_image(self, prompt: str, 
                            size: str = "512x512") -> Optional[bytes]:
        """Generate an image from a prompt"""
        session = await self.get_session()
        
        payload = {
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json"
        }
        
        try:
            async with session.post(
                f"{self.api_url}/v1/images/generations",
                json=payload,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    b64_data = data.get("data", [{}])[0].get("b64_json", "")
                    if b64_data:
                        return base64.b64decode(b64_data)
                return None
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None
    
    async def generate_code(self, prompt: str, 
                           language: str = "python") -> str:
        """Generate code from a prompt"""
        full_prompt = f"Generate {language} code for: {prompt}\n\nOnly output the code, no explanations."
        return await self.chat(full_prompt)


def create_discord_bot(config: BotConfig) -> Optional[commands.Bot]:
    """Create and configure the Discord bot"""
    
    if not HAS_DISCORD:
        logger.error("discord.py not installed. Run: pip install discord.py")
        return None
    
    if not HAS_AIOHTTP:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return None
    
    # Create bot with intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    
    bot = commands.Bot(
        command_prefix=config.command_prefix,
        intents=intents,
        description="Enigma AI Engine Discord Bot"
    )
    
    # Store config and AI client
    bot.forge_config = config
    bot.forge_client = Enigma AI EngineClient(config.forge_api_url, config.forge_api_key)
    bot.conversation_contexts: Dict[str, str] = {}  # user_id -> conversation_id
    bot.rate_limits: Dict[int, List[float]] = {}  # user_id -> timestamps
    
    # ==================== Events ====================
    
    @bot.event
    async def on_ready():
        """Called when bot is ready"""
        logger.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
        logger.info(f"Connected to {len(bot.guilds)} guilds")
        
        # Sync slash commands
        try:
            synced = await bot.tree.sync()
            logger.info(f"Synced {len(synced)} slash commands")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
        
        # Set status
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="/chat"
            )
        )
    
    @bot.event
    async def on_message(message: discord.Message):
        """Handle messages"""
        # Ignore bot's own messages
        if message.author == bot.user:
            return
        
        # Process commands
        await bot.process_commands(message)
        
        # Respond to mentions
        if bot.user in message.mentions:
            # Remove mention from message
            content = message.content.replace(f"<@{bot.user.id}>", "").strip()
            if content:
                async with message.channel.typing():
                    response = await bot.forge_client.chat(content)
                    # Split long responses
                    for chunk in split_message(response, config.max_response_length):
                        await message.reply(chunk)
    
    # ==================== Slash Commands ====================
    
    @bot.tree.command(name="chat", description="Chat with Enigma AI Engine")
    @app_commands.describe(message="Your message to the AI")
    async def chat_command(interaction: discord.Interaction, message: str):
        """Chat with the AI"""
        await interaction.response.defer(thinking=True)
        
        # Get user's conversation context
        user_key = str(interaction.user.id)
        conv_id = bot.conversation_contexts.get(user_key)
        
        response = await bot.forge_client.chat(message, conversation_id=conv_id)
        
        # Store conversation ID if returned
        # (depends on API implementation)
        
        for chunk in split_message(response, config.max_response_length):
            if chunk == split_message(response, config.max_response_length)[0]:
                await interaction.followup.send(chunk)
            else:
                await interaction.channel.send(chunk)
    
    @bot.tree.command(name="imagine", description="Generate an image")
    @app_commands.describe(prompt="Description of the image to generate")
    async def imagine_command(interaction: discord.Interaction, prompt: str):
        """Generate an image"""
        await interaction.response.defer(thinking=True)
        
        image_data = await bot.forge_client.generate_image(prompt)
        
        if image_data:
            file = discord.File(io.BytesIO(image_data), filename="generated.png")
            embed = discord.Embed(title="Generated Image", description=prompt)
            embed.set_image(url="attachment://generated.png")
            await interaction.followup.send(embed=embed, file=file)
        else:
            await interaction.followup.send("Failed to generate image. Please try again.")
    
    @bot.tree.command(name="code", description="Generate code")
    @app_commands.describe(
        prompt="Description of the code to generate",
        language="Programming language (default: python)"
    )
    @app_commands.choices(language=[
        app_commands.Choice(name="Python", value="python"),
        app_commands.Choice(name="JavaScript", value="javascript"),
        app_commands.Choice(name="TypeScript", value="typescript"),
        app_commands.Choice(name="Rust", value="rust"),
        app_commands.Choice(name="Go", value="go"),
        app_commands.Choice(name="C++", value="cpp"),
        app_commands.Choice(name="Java", value="java"),
    ])
    async def code_command(interaction: discord.Interaction, 
                          prompt: str, 
                          language: str = "python"):
        """Generate code"""
        await interaction.response.defer(thinking=True)
        
        code = await bot.forge_client.generate_code(prompt, language)
        
        # Format as code block
        formatted = f"```{language}\n{code}\n```"
        
        for chunk in split_message(formatted, config.max_response_length):
            if chunk == split_message(formatted, config.max_response_length)[0]:
                await interaction.followup.send(chunk)
            else:
                await interaction.channel.send(chunk)
    
    @bot.tree.command(name="clear", description="Clear your conversation history")
    async def clear_command(interaction: discord.Interaction):
        """Clear conversation context"""
        user_key = str(interaction.user.id)
        if user_key in bot.conversation_contexts:
            del bot.conversation_contexts[user_key]
        await interaction.response.send_message("Conversation history cleared!", ephemeral=True)
    
    @bot.tree.command(name="help", description="Show help information")
    async def help_command(interaction: discord.Interaction):
        """Show help"""
        embed = discord.Embed(
            title="Enigma AI Engine Bot Help",
            description="AI-powered Discord bot",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="/chat",
            value="Chat with the AI assistant",
            inline=False
        )
        embed.add_field(
            name="/imagine",
            value="Generate an image from a description",
            inline=False
        )
        embed.add_field(
            name="/code",
            value="Generate code in various languages",
            inline=False
        )
        embed.add_field(
            name="/clear",
            value="Clear your conversation history",
            inline=False
        )
        embed.add_field(
            name="@mention",
            value="You can also mention the bot to chat",
            inline=False
        )
        
        await interaction.response.send_message(embed=embed)
    
    # ==================== Text Commands (legacy) ====================
    
    @bot.command(name="ask")
    async def ask_command(ctx: commands.Context, *, question: str):
        """Ask the AI a question"""
        async with ctx.typing():
            response = await bot.forge_client.chat(question)
            for chunk in split_message(response, config.max_response_length):
                await ctx.reply(chunk)
    
    return bot


def split_message(text: str, max_length: int = 2000) -> List[str]:
    """Split a message into chunks that fit Discord's limit"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current = ""
    
    for line in text.split('\n'):
        if len(current) + len(line) + 1 <= max_length:
            current += line + '\n'
        else:
            if current:
                chunks.append(current.strip())
            current = line + '\n'
    
    if current:
        chunks.append(current.strip())
    
    return chunks if chunks else [text[:max_length]]


async def run_bot(config: BotConfig):
    """Run the Discord bot"""
    bot = create_discord_bot(config)
    if bot:
        try:
            await bot.start(config.token)
        finally:
            await bot.forge_client.close()


def main():
    """Main entry point"""
    # Load config from environment
    config = BotConfig(
        token=os.environ.get("DISCORD_TOKEN", ""),
        forge_api_url=os.environ.get("FORGE_API_URL", "http://localhost:8000"),
        forge_api_key=os.environ.get("FORGE_API_KEY", "")
    )
    
    if not config.token:
        print("Error: DISCORD_TOKEN environment variable not set")
        print("Get your token from https://discord.com/developers/applications")
        return
    
    print("Starting Enigma AI Engine Discord Bot...")
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
