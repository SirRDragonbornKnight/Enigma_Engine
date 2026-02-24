"""
Echo Brick - Example brick for Enigma AI Engine

This is a simple example brick that demonstrates:
1. How to structure a brick
2. How to connect to the router
3. How to handle commands
4. How to send responses

USE THIS AS A TEMPLATE for creating new bricks!

How it works:
- The brick connects TO the router (bricks are clients, router is server)
- On connect, brick sends a registration message
- Router acknowledges registration
- Brick then waits for commands from router
- When a command arrives, brick processes it and sends response

Protocol:
- All messages use 4-byte length prefix (big-endian) + JSON payload
- Registration: {"type": "register", "brick_id": "...", "name": "...", "capabilities": [...]}
- Commands arrive as: {"id": "...", "type": "command", "data": {"command": "...", "args": {...}}}
- Responses: {"id": "...", "type": "response", "success": true/false, "data": {...}}

Usage:
    # Start router first (from GUI or command line)
    # Then run this brick:
    python -m bricks.echo.main
"""

import asyncio
import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EchoBrick:
    """
    Example brick that demonstrates the brick architecture.
    
    This brick provides simple text manipulation commands:
    - echo: Return the message unchanged
    - reverse: Reverse a string
    - count: Count words in text
    """
    
    def __init__(self, config_path: Path = None):
        """
        Initialize the brick from its config file.
        
        Args:
            config_path: Path to brick.json. Defaults to same directory as this file.
        """
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "brick.json"
        
        with open(config_path, encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Basic brick info
        self.brick_id = self.config.get("id", "echo")
        self.name = self.config.get("name", "Echo Brick")
        self.running = False
        
        # Router connection info
        self.router_host = "127.0.0.1"
        self.router_port = 9900  # Router always on 9900
        
        # Connection state
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        
        # Get capabilities from commands list
        self.capabilities = [
            cmd.get("name", "") 
            for cmd in self.config.get("commands", [])
        ]
        
        logger.info(f"Initialized brick: {self.name} ({self.brick_id})")
        logger.info(f"Capabilities: {self.capabilities}")
    
    # =========================================================================
    # NETWORK PROTOCOL
    # =========================================================================
    
    async def connect(self) -> bool:
        """
        Connect to the router.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.router_host, self.router_port
            )
            logger.info(f"Connected to router at {self.router_host}:{self.router_port}")
            return True
        except ConnectionRefusedError:
            logger.error("Connection refused. Is the router running?")
            logger.error("Start the router from the GUI's Bricks tab first.")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def send_message(self, data: Dict) -> bool:
        """
        Send a message to the router.
        
        Protocol: 4-byte length prefix (big-endian) + JSON payload
        
        Args:
            data: Dictionary to send as JSON
            
        Returns:
            True if sent, False on error
        """
        if not self.writer:
            return False
        
        try:
            # Encode JSON
            msg = json.dumps(data).encode('utf-8')
            # Create length prefix (4 bytes, big-endian)
            length = struct.pack('>I', len(msg))
            # Send length + message
            self.writer.write(length + msg)
            await self.writer.drain()
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict]:
        """
        Receive a message from the router.
        
        Protocol: 4-byte length prefix (big-endian) + JSON payload
        
        Returns:
            Parsed JSON dict, or None on error/disconnect
        """
        if not self.reader:
            return None
        
        try:
            # Read 4-byte length prefix
            length_data = await self.reader.readexactly(4)
            length = struct.unpack('>I', length_data)[0]
            
            # Sanity check
            if length > 1_000_000:  # 1MB max
                logger.warning(f"Message too large: {length}")
                return None
            
            # Read message body
            data = await self.reader.readexactly(length)
            return json.loads(data.decode('utf-8'))
        
        except asyncio.IncompleteReadError:
            # Connection closed
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"Receive error: {e}")
            return None
    
    async def register(self) -> bool:
        """
        Register this brick with the router.
        
        Sends: {"type": "register", "brick_id": "...", "name": "...", "capabilities": [...]}
        Expects: {"type": "registered", "brick_id": "...", "status": "ok"}
        
        Returns:
            True if registered successfully
        """
        register_msg = {
            "type": "register",
            "brick_id": self.brick_id,
            "name": self.name,
            "capabilities": self.capabilities
        }
        
        if not await self.send_message(register_msg):
            return False
        
        # Wait for acknowledgment
        response = await self.receive_message()
        if response and response.get("type") == "registered":
            logger.info(f"[OK] Registered with router as '{self.brick_id}'")
            return True
        
        logger.error(f"Registration failed: {response}")
        return False
    
    # =========================================================================
    # COMMAND HANDLERS
    # Add your custom commands here!
    # =========================================================================
    
    def cmd_echo(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Echo command - returns the message unchanged.
        
        Args:
            args: {"message": "text to echo"}
            
        Returns:
            {"message": "text to echo"}
        """
        message = args.get("message", "")
        logger.info(f"Echo: {message}")
        return {"message": message}
    
    def cmd_reverse(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverse command - reverses a string.
        
        Args:
            args: {"text": "hello"}
            
        Returns:
            {"result": "olleh", "original": "hello"}
        """
        text = args.get("text", "")
        reversed_text = text[::-1]
        logger.info(f"Reverse: '{text}' -> '{reversed_text}'")
        return {"result": reversed_text, "original": text}
    
    def cmd_count(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Count command - counts words in text.
        
        Args:
            args: {"text": "hello world"}
            
        Returns:
            {"characters": 11, "words": 2, "lines": 1}
        """
        text = args.get("text", "")
        chars = len(text)
        words = len(text.split())
        lines = len(text.splitlines()) if text else 0
        lines = max(1, lines)  # At least 1 line if text exists
        logger.info(f"Count: {chars} chars, {words} words, {lines} lines")
        return {"characters": chars, "words": words, "lines": lines}
    
    def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Status command - returns brick status.
        
        Returns:
            {"name": "...", "brick_id": "...", "running": true}
        """
        return {
            "name": self.name,
            "brick_id": self.brick_id,
            "running": self.running
        }
    
    def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stop command - stops the brick.
        
        Returns:
            {"stopped": true}
        """
        logger.info("Stop command received")
        self.running = False
        return {"stopped": True}
    
    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming message from the router.
        
        Message format:
            {"id": "msg-123", "type": "command", "data": {"command": "echo", "args": {...}}}
        
        Response format:
            {"id": "msg-123", "type": "response", "success": true, "data": {...}}
        
        Args:
            message: Incoming message dict
            
        Returns:
            Response dict to send back
        """
        msg_type = message.get("type", "")
        msg_id = message.get("id", "")
        data = message.get("data", {})
        
        # Handle command messages
        if msg_type == "command":
            command = data.get("command", "")
            args = data.get("args", {})
            
            # Find handler method (cmd_<command>)
            handler = getattr(self, f"cmd_{command}", None)
            
            if handler:
                try:
                    # Call the handler
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(args)
                    else:
                        result = handler(args)
                    
                    return {
                        "id": msg_id,
                        "type": "response",
                        "success": True,
                        "data": result
                    }
                except Exception as e:
                    logger.exception(f"Command '{command}' failed")
                    return {
                        "id": msg_id,
                        "type": "error",
                        "success": False,
                        "error": str(e)
                    }
            else:
                return {
                    "id": msg_id,
                    "type": "error",
                    "success": False,
                    "error": f"Unknown command: {command}"
                }
        
        # Handle ping messages
        elif msg_type == "ping":
            return {"id": msg_id, "type": "pong"}
        
        # Unknown message type
        return {
            "id": msg_id,
            "type": "error",
            "success": False,
            "error": f"Unknown message type: {msg_type}"
        }
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    async def run(self):
        """
        Main loop: connect to router and handle messages.
        
        1. Connect to router
        2. Register this brick
        3. Wait for commands and respond
        4. Clean up on exit
        """
        self.running = True
        
        # Step 1: Connect to router
        if not await self.connect():
            logger.error("Could not connect to router.")
            logger.error("Make sure the router is running (start from GUI's Bricks tab)")
            return
        
        # Step 2: Register with router
        if not await self.register():
            logger.error("Registration failed")
            return
        
        logger.info(f"[OK] Brick '{self.name}' is ready!")
        logger.info("Waiting for commands...")
        
        # Step 3: Message loop
        try:
            while self.running:
                message = await self.receive_message()
                if message is None:
                    # Connection lost
                    logger.warning("Connection to router lost")
                    break
                
                logger.debug(f"Received: {message}")
                
                # Handle message and send response
                response = await self.handle_message(message)
                await self.send_message(response)
        
        except asyncio.CancelledError:
            logger.info("Brick cancelled")
        except Exception as e:
            logger.exception(f"Brick error: {e}")
        finally:
            # Step 4: Cleanup
            if self.writer:
                self.writer.close()
                try:
                    await self.writer.wait_closed()
                except Exception:
                    pass
            logger.info("Brick stopped")


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    """Main entry point."""
    brick = EchoBrick()
    await brick.run()


if __name__ == "__main__":
    print("=" * 50)
    print("  Echo Brick - Example Brick for Enigma Engine")
    print("=" * 50)
    print()
    print("This brick will connect to the router on port 9900.")
    print("Make sure the router is running first!")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBrick stopped by user")
