"""
Conversation Export Utility

Export conversations to various formats: JSON, Markdown, PDF, HTML.

Usage:
    from enigma_engine.memory.export import ConversationExporter
    
    exporter = ConversationExporter()
    
    # Export to Markdown
    exporter.to_markdown(conversation, "chat.md")
    
    # Export to JSON
    exporter.to_json(conversation, "chat.json")
    
    # Export to HTML
    exporter.to_html(conversation, "chat.html")
"""

import html
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Conversation:
    """A conversation with messages."""
    id: str
    title: str = "Untitled Conversation"
    messages: Optional[list[Message]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        messages = [
            Message(**m) if isinstance(m, dict) else m
            for m in data.get("messages", [])
        ]
        return cls(
            id=data.get("id", "unknown"),
            title=data.get("title", "Untitled"),
            messages=messages,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {})
        )


class ConversationExporter:
    """Export conversations to various formats."""
    
    def __init__(self):
        """Initialize exporter."""
    
    def to_json(
        self,
        conversation: Union[Conversation, dict, list],
        output_path: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Export conversation to JSON.
        
        Args:
            conversation: Conversation object, dict, or list of messages
            output_path: Optional file path to save to
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        # Normalize input
        if isinstance(conversation, Conversation):
            data = conversation.to_dict()
        elif isinstance(conversation, list):
            data = {
                "messages": [
                    m.to_dict() if isinstance(m, Message) else m
                    for m in conversation
                ]
            }
        else:
            data = conversation
        
        json_str = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
        
        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
            logger.info(f"Exported conversation to {output_path}")
        
        return json_str
    
    def to_markdown(
        self,
        conversation: Union[Conversation, dict, list],
        output_path: Optional[str] = None,
        include_metadata: bool = False,
        include_timestamps: bool = True
    ) -> str:
        """
        Export conversation to Markdown.
        
        Args:
            conversation: Conversation object, dict, or list of messages
            output_path: Optional file path to save to
            include_metadata: Whether to include metadata
            include_timestamps: Whether to include timestamps
            
        Returns:
            Markdown string
        """
        # Normalize input
        if isinstance(conversation, dict):
            conversation = Conversation.from_dict(conversation)
        elif isinstance(conversation, list):
            conversation = Conversation(
                id="export",
                title="Conversation Export",
                messages=[
                    Message(**m) if isinstance(m, dict) else m
                    for m in conversation
                ]
            )
        
        lines = []
        
        # Header
        lines.append(f"# {conversation.title}")
        lines.append("")
        
        if include_metadata:
            lines.append(f"**ID:** {conversation.id}")
            lines.append(f"**Created:** {conversation.created_at}")
            if conversation.metadata:
                lines.append(f"**Metadata:** {json.dumps(conversation.metadata)}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Messages
        for msg in conversation.messages:
            role_display = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg.role, msg.role.title())
            
            lines.append(f"### {role_display}")
            
            if include_timestamps and msg.timestamp:
                lines.append(f"*{msg.timestamp}*")
            
            lines.append("")
            lines.append(msg.content)
            lines.append("")
        
        markdown = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")
            logger.info(f"Exported conversation to {output_path}")
        
        return markdown
    
    def to_html(
        self,
        conversation: Union[Conversation, dict, list],
        output_path: Optional[str] = None,
        include_styles: bool = True,
        dark_mode: bool = False
    ) -> str:
        """
        Export conversation to HTML.
        
        Args:
            conversation: Conversation object, dict, or list of messages
            output_path: Optional file path to save to
            include_styles: Whether to include CSS styles
            dark_mode: Use dark mode colors
            
        Returns:
            HTML string
        """
        # Normalize input
        if isinstance(conversation, dict):
            conversation = Conversation.from_dict(conversation)
        elif isinstance(conversation, list):
            conversation = Conversation(
                id="export",
                title="Conversation Export",
                messages=[
                    Message(**m) if isinstance(m, dict) else m
                    for m in conversation
                ]
            )
        
        # Build HTML
        parts = ['<!DOCTYPE html>', '<html lang="en">', '<head>',
                 '<meta charset="UTF-8">',
                 '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
                 f'<title>{html.escape(conversation.title)}</title>']
        
        if include_styles:
            bg_color = "#1e1e1e" if dark_mode else "#ffffff"
            text_color = "#d4d4d4" if dark_mode else "#333333"
            user_bg = "#2d4a3e" if dark_mode else "#e3f2fd"
            assistant_bg = "#3d3d3d" if dark_mode else "#f5f5f5"
            system_bg = "#4a3d2d" if dark_mode else "#fff3e0"
            
            parts.append(f'''<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background-color: {bg_color};
    color: {text_color};
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
}}
h1 {{
    border-bottom: 2px solid #ccc;
    padding-bottom: 10px;
}}
.message {{
    margin: 15px 0;
    padding: 15px;
    border-radius: 10px;
}}
.message-user {{
    background-color: {user_bg};
    margin-left: 20px;
}}
.message-assistant {{
    background-color: {assistant_bg};
    margin-right: 20px;
}}
.message-system {{
    background-color: {system_bg};
    font-style: italic;
}}
.role {{
    font-weight: bold;
    margin-bottom: 5px;
    color: {"#aaa" if dark_mode else "#666"};
}}
.timestamp {{
    font-size: 0.8em;
    color: {"#888" if dark_mode else "#999"};
}}
.content {{
    white-space: pre-wrap;
    word-wrap: break-word;
}}
.metadata {{
    font-size: 0.9em;
    color: {"#888" if dark_mode else "#666"};
    margin-bottom: 20px;
}}
</style>''')
        
        parts.extend(['</head>', '<body>',
                     f'<h1>{html.escape(conversation.title)}</h1>',
                     f'<div class="metadata">ID: {html.escape(conversation.id)} | Created: {html.escape(str(conversation.created_at))}</div>'])
        
        # Messages
        for msg in conversation.messages:
            role_display = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg.role, msg.role.title())
            
            role_class = f"message-{msg.role}"
            
            parts.append(f'<div class="message {role_class}">')
            parts.append(f'<div class="role">{html.escape(role_display)}</div>')
            
            if msg.timestamp:
                parts.append(f'<div class="timestamp">{html.escape(msg.timestamp)}</div>')
            
            # Escape content but preserve basic formatting
            content = html.escape(msg.content)
            # Convert code blocks
            content = self._format_code_blocks(content)
            
            parts.append(f'<div class="content">{content}</div>')
            parts.append('</div>')
        
        parts.extend(['</body>', '</html>'])
        
        html_str = "\n".join(parts)
        
        if output_path:
            Path(output_path).write_text(html_str, encoding="utf-8")
            logger.info(f"Exported conversation to {output_path}")
        
        return html_str
    
    def _format_code_blocks(self, text: str) -> str:
        """Format code blocks in HTML."""
        import re

        # Triple backtick code blocks
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            return f'<pre><code class="language-{lang}">{code}</code></pre>'
        
        text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
        
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        return text
    
    def to_text(
        self,
        conversation: Union[Conversation, dict, list],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export conversation to plain text.
        
        Args:
            conversation: Conversation object, dict, or list of messages
            output_path: Optional file path to save to
            
        Returns:
            Plain text string
        """
        # Normalize input
        if isinstance(conversation, dict):
            conversation = Conversation.from_dict(conversation)
        elif isinstance(conversation, list):
            conversation = Conversation(
                id="export",
                title="Conversation Export",
                messages=[
                    Message(**m) if isinstance(m, dict) else m
                    for m in conversation
                ]
            )
        
        lines = []
        lines.append(conversation.title)
        lines.append("=" * len(conversation.title))
        lines.append("")
        
        for msg in conversation.messages:
            role_display = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg.role, msg.role.title())
            
            lines.append(f"[{role_display}]")
            lines.append(msg.content)
            lines.append("")
        
        text = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(text, encoding="utf-8")
            logger.info(f"Exported conversation to {output_path}")
        
        return text


def export_conversation(
    conversation: Union[Conversation, dict, list],
    output_path: str,
    format: str = "auto"
) -> str:
    """
    Export a conversation to a file.
    
    Args:
        conversation: Conversation to export
        output_path: Output file path
        format: Format - "json", "markdown", "html", "text", or "auto" (detect from extension)
        
    Returns:
        The exported content
    """
    exporter = ConversationExporter()
    
    # Auto-detect format from extension
    if format == "auto":
        ext = Path(output_path).suffix.lower()
        format = {
            ".json": "json",
            ".md": "markdown",
            ".markdown": "markdown", 
            ".html": "html",
            ".htm": "html",
            ".txt": "text",
        }.get(ext, "json")
    
    if format == "json":
        return exporter.to_json(conversation, output_path)
    elif format == "markdown":
        return exporter.to_markdown(conversation, output_path)
    elif format == "html":
        return exporter.to_html(conversation, output_path)
    elif format == "text":
        return exporter.to_text(conversation, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
