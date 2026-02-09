"""
CLI Chat Mode for Enigma AI Engine

Pure terminal interface for SSH/headless operation.

Usage:
    python -m enigma_engine.cli.chat [--model MODEL_PATH]
    
    # Or programmatically
    from enigma_engine.cli.chat import CLIChat
    chat = CLIChat(model_path="models/my_model")
    chat.run()
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Try to import readline for better input handling
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Background
    BG_BLUE = "\033[44m"
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, "")


class CLIChat:
    """
    Command-line chat interface.
    
    Features:
    - Pure text interface
    - Command history (with readline)
    - Slash commands
    - Session saving
    - Works over SSH
    """
    
    COMMANDS = {
        "/help": "Show available commands",
        "/clear": "Clear conversation history",
        "/save": "Save conversation to file",
        "/load": "Load model from path",
        "/info": "Show model information",
        "/config": "Show/set generation config",
        "/system": "Set system prompt",
        "/history": "Show conversation history",
        "/quit": "Exit chat (or use Ctrl+C)"
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        no_color: bool = False,
        system_prompt: Optional[str] = None
    ):
        if no_color or not sys.stdout.isatty():
            Colors.disable()
        
        self.model_path = model_path
        self.engine = None
        self.history: List[dict] = []
        self._max_history = 200  # Prevent unbounded memory growth
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        
        # Generation config
        self.config = {
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # History file for readline
        self.history_file = Path.home() / ".enigma_cli_history"
        
        self._load_readline_history()
    
    def _load_readline_history(self):
        """Load command history from file."""
        if HAS_READLINE and self.history_file.exists():
            try:
                readline.read_history_file(str(self.history_file))
            except Exception:
                pass
    
    def _save_readline_history(self):
        """Save command history to file."""
        if HAS_READLINE:
            try:
                readline.write_history_file(str(self.history_file))
            except Exception:
                pass
    
    def _print_header(self):
        """Print welcome header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}Enigma AI Engine - CLI Chat{Colors.RESET}")
        print(f"{Colors.DIM}Type /help for commands, /quit to exit{Colors.RESET}")
        print()
    
    def _print_help(self):
        """Print command help."""
        print(f"\n{Colors.BOLD}Available Commands:{Colors.RESET}")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {Colors.CYAN}{cmd:12}{Colors.RESET} {desc}")
        print()
    
    def _prompt(self) -> str:
        """Get user input with styled prompt."""
        try:
            return input(f"{Colors.GREEN}{Colors.BOLD}You:{Colors.RESET} ")
        except EOFError:
            return "/quit"
    
    def _print_response(self, text: str):
        """Print AI response with styling."""
        print(f"{Colors.BLUE}{Colors.BOLD}AI:{Colors.RESET} {text}")
    
    def _print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.RED}Error: {text}{Colors.RESET}")
    
    def _print_info(self, text: str):
        """Print info message."""
        print(f"{Colors.DIM}{text}{Colors.RESET}")
    
    def load_model(self, path: Optional[str] = None):
        """Load the AI model."""
        path = path or self.model_path
        
        if not path:
            self._print_error("No model path specified")
            return False
        
        try:
            from ..core.inference import EnigmaEngine
            
            self._print_info(f"Loading model from {path}...")
            self.engine = EnigmaEngine(path)
            self.model_path = path
            self._print_info(f"Model loaded successfully")
            return True
        except Exception as e:
            self._print_error(f"Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """Generate a response."""
        if not self.engine:
            return "[No model loaded. Use /load <path> to load a model]"
        
        try:
            # Build context from history
            context = f"System: {self.system_prompt}\n\n"
            for msg in self.history[-10:]:  # Last 10 messages
                role = "User" if msg["role"] == "user" else "AI"
                context += f"{role}: {msg['content']}\n"
            context += f"User: {prompt}\nAI:"
            
            response = self.engine.generate(
                context,
                max_gen=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"]
            )
            
            return response.strip()
        except Exception as e:
            return f"[Error: {e}]"
    
    def handle_command(self, cmd: str) -> bool:
        """
        Handle a slash command.
        
        Returns:
            True to continue, False to exit
        """
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "/quit" or command == "/exit":
            return False
        
        elif command == "/help":
            self._print_help()
        
        elif command == "/clear":
            self.history.clear()
            self._print_info("Conversation cleared")
        
        elif command == "/save":
            filename = args or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump({
                        "system_prompt": self.system_prompt,
                        "messages": self.history,
                        "saved_at": datetime.now().isoformat()
                    }, f, indent=2)
                self._print_info(f"Saved to {filename}")
            except Exception as e:
                self._print_error(f"Failed to save: {e}")
        
        elif command == "/load":
            if args:
                self.load_model(args)
            else:
                self._print_error("Usage: /load <model_path>")
        
        elif command == "/info":
            if self.engine:
                print(f"\n{Colors.BOLD}Model Info:{Colors.RESET}")
                print(f"  Path: {self.model_path}")
                if hasattr(self.engine, 'model'):
                    params = sum(p.numel() for p in self.engine.model.parameters())
                    print(f"  Parameters: {params:,}")
                print()
            else:
                self._print_info("No model loaded")
        
        elif command == "/config":
            if args:
                # Parse key=value
                try:
                    key, value = args.split("=")
                    key = key.strip()
                    value = float(value.strip())
                    if key in self.config:
                        self.config[key] = value if key != "max_tokens" else int(value)
                        self._print_info(f"Set {key} = {self.config[key]}")
                    else:
                        self._print_error(f"Unknown config: {key}")
                except ValueError:
                    self._print_error("Usage: /config key=value")
            else:
                print(f"\n{Colors.BOLD}Generation Config:{Colors.RESET}")
                for key, value in self.config.items():
                    print(f"  {key}: {value}")
                print(f"\n{Colors.DIM}Use /config key=value to change{Colors.RESET}\n")
        
        elif command == "/system":
            if args:
                self.system_prompt = args
                self._print_info(f"System prompt updated")
            else:
                print(f"\n{Colors.BOLD}System Prompt:{Colors.RESET}")
                print(f"  {self.system_prompt}\n")
        
        elif command == "/history":
            if self.history:
                print(f"\n{Colors.BOLD}Conversation History:{Colors.RESET}")
                for i, msg in enumerate(self.history):
                    role = "You" if msg["role"] == "user" else "AI"
                    content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                    print(f"  {i+1}. {role}: {content}")
                print()
            else:
                self._print_info("No history yet")
        
        else:
            self._print_error(f"Unknown command: {command}")
        
        return True
    
    def run(self):
        """Run the chat loop."""
        self._print_header()
        
        # Load model if specified
        if self.model_path:
            self.load_model()
        else:
            self._print_info("No model loaded. Use /load <path> to load a model.")
        
        print()
        
        try:
            while True:
                try:
                    user_input = self._prompt().strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith("/"):
                        if not self.handle_command(user_input):
                            break
                        continue
                    
                    # Add to history
                    self.history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Generate response
                    response = self.generate(user_input)
                    self._print_response(response)
                    print()
                    
                    # Add to history
                    self.history.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Trim history to prevent unbounded growth
                    if len(self.history) > self._max_history:
                        self.history = self.history[-self._max_history:]
                    
                except KeyboardInterrupt:
                    print()  # New line after ^C
                    self._print_info("Use /quit to exit")
                    continue
                
        finally:
            self._save_readline_history()
            print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}\n")


def main():
    """Main entry point for CLI chat."""
    parser = argparse.ArgumentParser(
        description="Enigma AI Engine - CLI Chat Interface"
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to model directory"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--system", "-s",
        help="System prompt for the AI"
    )
    
    args = parser.parse_args()
    
    chat = CLIChat(
        model_path=args.model,
        no_color=args.no_color,
        system_prompt=args.system
    )
    chat.run()


if __name__ == "__main__":
    main()
