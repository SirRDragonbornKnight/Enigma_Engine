"""
Slack Bot Template for Enigma AI Engine

A ready-to-deploy Slack bot that integrates with Enigma AI Engine.

Features:
- Chat with AI in channels or DMs
- Slash commands for various AI tasks
- Image generation
- Code generation
- Thread conversations

Usage:
    1. Create a Slack app at https://api.slack.com/apps
    2. Enable Socket Mode and get the app token
    3. Add OAuth scopes: chat:write, commands, app_mentions:read, im:history
    4. Set environment variables:
       - SLACK_BOT_TOKEN
       - SLACK_APP_TOKEN
       - FORGE_API_URL
    5. Run: python slack_bot.py
"""

import asyncio
import base64
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional imports
try:
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    from slack_sdk.web.async_client import AsyncWebClient
    HAS_SLACK = True
except ImportError:
    HAS_SLACK = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SlackBotConfig:
    """Slack bot configuration"""
    bot_token: str
    app_token: str
    forge_api_url: str = "http://localhost:8000"
    forge_api_key: str = ""
    max_response_length: int = 3000


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
    
    async def summarize(self, text: str) -> str:
        """Summarize text"""
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return await self.chat(prompt)


def create_slack_bot(config: SlackBotConfig) -> Optional['App']:
    """Create and configure the Slack bot"""
    
    if not HAS_SLACK:
        logger.error("slack_bolt not installed. Run: pip install slack_bolt")
        return None
    
    if not HAS_AIOHTTP:
        logger.error("aiohttp not installed. Run: pip install aiohttp")
        return None
    
    # Create the app
    app = App(token=config.bot_token)
    
    # Store config and AI client
    app.forge_config = config
    app.forge_client = Enigma AI EngineClient(config.forge_api_url, config.forge_api_key)
    app.thread_contexts: Dict[str, str] = {}  # thread_ts -> conversation_id
    
    # ==================== Message Handlers ====================
    
    @app.event("app_mention")
    def handle_mention(event, say, client):
        """Handle @mentions of the bot"""
        text = event.get("text", "")
        user = event.get("user")
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        # Remove the mention from the text
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        if not text:
            say("Hi! How can I help you?", thread_ts=thread_ts)
            return
        
        # Get response from Enigma AI Engine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(app.forge_client.chat(text))
        finally:
            loop.close()
        
        # Split long responses
        for chunk in split_message(response, config.max_response_length):
            say(chunk, thread_ts=thread_ts)
    
    @app.event("message")
    def handle_direct_message(event, say, client):
        """Handle direct messages"""
        # Only respond in DMs
        channel_type = event.get("channel_type")
        if channel_type != "im":
            return
        
        # Ignore bot messages
        if event.get("bot_id"):
            return
        
        text = event.get("text", "")
        if not text:
            return
        
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        # Get response from Enigma AI Engine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            conv_id = app.thread_contexts.get(thread_ts)
            response = loop.run_until_complete(
                app.forge_client.chat(text, conversation_id=conv_id)
            )
        finally:
            loop.close()
        
        for chunk in split_message(response, config.max_response_length):
            say(chunk, thread_ts=thread_ts)
    
    # ==================== Slash Commands ====================
    
    @app.command("/forge")
    def handle_forge_command(ack, command, respond):
        """Handle /forge command"""
        ack()
        
        text = command.get("text", "").strip()
        if not text:
            respond("Usage: /forge <your message>")
            return
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(app.forge_client.chat(text))
        finally:
            loop.close()
        
        for chunk in split_message(response, config.max_response_length):
            respond(chunk)
    
    @app.command("/imagine")
    def handle_imagine_command(ack, command, respond, client):
        """Handle /imagine command for image generation"""
        ack()
        
        prompt = command.get("text", "").strip()
        if not prompt:
            respond("Usage: /imagine <description of the image>")
            return
        
        respond("Generating image... This may take a moment.")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            image_data = loop.run_until_complete(
                app.forge_client.generate_image(prompt)
            )
        finally:
            loop.close()
        
        if image_data:
            # Upload image to Slack
            try:
                result = client.files_upload_v2(
                    channel=command.get("channel_id"),
                    file=io.BytesIO(image_data),
                    filename="generated_image.png",
                    title=f"Generated: {prompt[:50]}..."
                )
                respond(f"Generated image for: _{prompt}_")
            except Exception as e:
                logger.error(f"Failed to upload image: {e}")
                respond("Generated the image but failed to upload it.")
        else:
            respond("Failed to generate image. Please try again.")
    
    @app.command("/code")
    def handle_code_command(ack, command, respond):
        """Handle /code command for code generation"""
        ack()
        
        text = command.get("text", "").strip()
        if not text:
            respond("Usage: /code [language] <description>\nExample: /code python function to sort a list")
            return
        
        # Parse language from the beginning
        parts = text.split(maxsplit=1)
        languages = ["python", "javascript", "typescript", "rust", "go", "java", "cpp", "c"]
        
        if parts[0].lower() in languages:
            language = parts[0].lower()
            prompt = parts[1] if len(parts) > 1 else ""
        else:
            language = "python"
            prompt = text
        
        if not prompt:
            respond("Please provide a description of the code you want.")
            return
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            code = loop.run_until_complete(
                app.forge_client.generate_code(prompt, language)
            )
        finally:
            loop.close()
        
        # Format as code block
        formatted = f"```{language}\n{code}\n```"
        
        for chunk in split_message(formatted, config.max_response_length):
            respond(chunk)
    
    @app.command("/summarize")
    def handle_summarize_command(ack, command, respond):
        """Handle /summarize command"""
        ack()
        
        text = command.get("text", "").strip()
        if not text:
            respond("Usage: /summarize <text to summarize>")
            return
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            summary = loop.run_until_complete(app.forge_client.summarize(text))
        finally:
            loop.close()
        
        respond(f"*Summary:*\n{summary}")
    
    @app.command("/forgehelp")
    def handle_help_command(ack, respond):
        """Handle /forgehelp command"""
        ack()
        
        help_text = """
*Enigma AI Engine Slack Bot Commands:*

- `/forge <message>` - Chat with the AI
- `/imagine <description>` - Generate an image
- `/code [language] <description>` - Generate code (default: Python)
- `/summarize <text>` - Summarize text
- `/forgehelp` - Show this help message

*Tips:*
- Mention @Enigma AI Engine in any channel to chat
- DM the bot for private conversations
- Start threads to maintain context
        """
        
        respond(help_text)
    
    # ==================== Shortcuts ====================
    
    @app.shortcut("summarize_message")
    def handle_summarize_shortcut(ack, shortcut, client, respond):
        """Handle message shortcut for summarization"""
        ack()
        
        # Get the message text
        message = shortcut.get("message", {})
        text = message.get("text", "")
        
        if not text:
            return
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            summary = loop.run_until_complete(app.forge_client.summarize(text))
        finally:
            loop.close()
        
        # Post summary as ephemeral message
        client.chat_postEphemeral(
            channel=shortcut.get("channel", {}).get("id"),
            user=shortcut.get("user", {}).get("id"),
            text=f"*Summary:*\n{summary}"
        )
    
    # ==================== Home Tab ====================
    
    @app.event("app_home_opened")
    def handle_home_opened(client, event):
        """Update the app home tab"""
        user_id = event.get("user")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Welcome to Enigma AI Engine"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Your AI assistant for Slack. Here's what I can do:"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Chat*\nMention me in any channel or send a DM to start a conversation."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Image Generation*\nUse `/imagine` to create images from descriptions."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Code Generation*\nUse `/code` to generate code in various languages."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Summarization*\nUse `/summarize` or the message shortcut to summarize text."
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Powered by Enigma AI Engine | Type `/forgehelp` for more commands"
                    }
                ]
            }
        ]
        
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": blocks
            }
        )
    
    return app


def split_message(text: str, max_length: int = 3000) -> List[str]:
    """Split a message into chunks that fit Slack's limit"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current = ""
    
    # Try to split on code blocks first
    in_code_block = False
    lines = text.split('\n')
    
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
        
        if len(current) + len(line) + 1 <= max_length:
            current += line + '\n'
        else:
            if current:
                # If we're in a code block, close it
                if in_code_block and not current.rstrip().endswith('```'):
                    current += '```\n'
                chunks.append(current.strip())
                
                # Start new chunk with code block if needed
                if in_code_block:
                    current = '```\n' + line + '\n'
                else:
                    current = line + '\n'
            else:
                current = line + '\n'
    
    if current:
        chunks.append(current.strip())
    
    return chunks if chunks else [text[:max_length]]


def run_bot(config: SlackBotConfig):
    """Run the Slack bot"""
    app = create_slack_bot(config)
    if app:
        handler = SocketModeHandler(app, config.app_token)
        logger.info("Starting Enigma AI Engine Slack Bot...")
        handler.start()


def main():
    """Main entry point"""
    # Load config from environment
    config = SlackBotConfig(
        bot_token=os.environ.get("SLACK_BOT_TOKEN", ""),
        app_token=os.environ.get("SLACK_APP_TOKEN", ""),
        forge_api_url=os.environ.get("FORGE_API_URL", "http://localhost:8000"),
        forge_api_key=os.environ.get("FORGE_API_KEY", "")
    )
    
    if not config.bot_token:
        print("Error: SLACK_BOT_TOKEN environment variable not set")
        return
    
    if not config.app_token:
        print("Error: SLACK_APP_TOKEN environment variable not set")
        print("Enable Socket Mode in your Slack app settings")
        return
    
    run_bot(config)


if __name__ == "__main__":
    main()
