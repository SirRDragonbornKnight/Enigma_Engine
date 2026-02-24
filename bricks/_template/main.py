"""
YOUR BRICK - Edit This File!

HOW TO CREATE YOUR BRICK:
1. Copy this folder (bricks/_template/) to bricks/yourbrick/
2. Edit brick.json - set your brick's id, name, commands, and ui widgets
3. Edit THIS FILE - implement your command handlers below

WHAT YOU EDIT:
    cmd_generate()     - Your main command logic
    cmd_*()            - Any other commands (match brick.json "commands")

WHAT YOU DON'T EDIT:
    brick_base.py      - Connection protocol (handled for you)
    connect/register   - Handled by base class

UI WIDGETS:
    Your brick's interface is defined in brick.json under "ui".widgets.
    The desktop GUI renders these automatically. Supported widget types:
        text_input  - Single-line text entry
        text_area   - Multi-line text box (set "rows" for height)
        number      - Numeric entry with default value
        button      - Sends a command (set "command" to match a cmd_* method)
        dropdown    - Selection from options list (set "options" and "default")
        checkbox    - Boolean toggle (set "default" to true/false)

    When a button is clicked, the GUI gathers all widget values by their
    "id" field and passes them as args to your cmd_* handler.
"""

import asyncio
import logging
from typing import Any, Dict

# Import the base class (DO NOT EDIT brick_base.py)
from brick_base import BrickClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# YOUR BRICK CLASS - EDIT THIS!
# =============================================================================

class MyBrick(BrickClient):
    """
    Your brick implementation. Rename this class to match your brick.

    Example: ImageGenBrick, AudioBrick, CodeBrick, etc.
    """

    # =========================================================================
    # REQUIRED: Your main command
    # =========================================================================
    
    async def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Your main command. Called when AI sends:
        [CMD]brick.send yourbrick generate prompt=...[/CMD]
        
        EDIT THIS to do what your brick actually does!
        
        Args:
            args: {"prompt": "...", ...} from AI or GUI
            
        Returns:
            Result dict sent back to AI
        """
        prompt = args.get("prompt", "")
        
        # =====================================================================
        # YOUR LOGIC HERE - replace this placeholder!
        # =====================================================================
        
        # Example: Send progress updates
        await self.send_update("progress", {"percent": 0, "message": "Starting..."})
        
        # Do your work here...
        await asyncio.sleep(0.5)  # Simulated work
        
        await self.send_update("progress", {"percent": 50, "message": "Processing..."})
        
        # More work...
        await asyncio.sleep(0.5)  # Simulated work
        
        await self.send_update("progress", {"percent": 100, "message": "Done!"})
        
        # Return result to AI
        return {
            "result": f"Generated from: {prompt}",
            "type": "text"  # or "image", "audio", etc.
        }
    
    # =========================================================================
    # OPTIONAL: Additional commands (match brick.json "commands")
    # =========================================================================
    
    async def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return brick status. Called by AI: [CMD]brick.status yourbrick[/CMD]"""
        return {
            "name": self.name,
            "brick_id": self.brick_id,
            "running": self.running,
            # Add your custom status info here
        }
    
    async def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the brick. Called by AI: [CMD]brick.stop yourbrick[/CMD]"""
        self.running = False
        return {"stopped": True}


# =============================================================================
# ENTRY POINT - runs your brick
# =============================================================================

async def main():
    """Run the brick."""
    brick = MyBrick()
    await brick.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Brick stopped by user")
