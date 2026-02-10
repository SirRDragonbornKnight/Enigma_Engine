"""
Data Exporter - Export data to various formats.

Features:
- Multiple export formats (JSON, CSV, Markdown, HTML)
- Template-based exports
- Batch exports
- Scheduled exports
- Compression support

Part of the Enigma AI Engine data management suite.
"""

import csv
import gzip
import html
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# EXPORT FORMATS
# =============================================================================

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    JSON_PRETTY = "json_pretty"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    YAML = "yaml"
    XML = "xml"


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"


# =============================================================================
# EXPORT OPTIONS
# =============================================================================

@dataclass
class ExportOptions:
    """Options for export."""
    format: ExportFormat = ExportFormat.JSON
    compression: CompressionType = CompressionType.NONE
    include_metadata: bool = True
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    indent: int = 2
    csv_delimiter: str = ","
    html_css: bool = True
    fields: Optional[list[str]] = None  # Specific fields to export
    exclude_fields: Optional[list[str]] = None
    filename_template: str = "{name}_{date}.{ext}"
    

# =============================================================================
# EXPORT RESULT
# =============================================================================

@dataclass
class ExportResult:
    """Result of an export."""
    success: bool
    format: ExportFormat
    size_bytes: int
    item_count: int
    file_path: Optional[Path] = None
    data: Optional[bytes] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


# =============================================================================
# FORMAT EXPORTERS
# =============================================================================

class JSONExporter:
    """Export to JSON format."""
    
    def export(
        self,
        data: Any,
        options: ExportOptions
    ) -> bytes:
        """Export data to JSON bytes."""
        indent = options.indent if options.format == ExportFormat.JSON_PRETTY else None
        
        result = {
            "data": data,
        }
        
        if options.include_metadata:
            result["metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "format": "json",
                "item_count": len(data) if isinstance(data, list) else 1
            }
        
        return json.dumps(result, indent=indent, default=str).encode("utf-8")


class CSVExporter:
    """Export to CSV format."""
    
    def export(
        self,
        data: list[dict],
        options: ExportOptions
    ) -> bytes:
        """Export data to CSV bytes."""
        if not data:
            return b""
        
        output = io.StringIO()
        
        # Determine fields
        if options.fields:
            fieldnames = options.fields
        else:
            # Collect all fields from all items
            fieldnames = set()
            for item in data:
                if isinstance(item, dict):
                    fieldnames.update(item.keys())
            fieldnames = sorted(fieldnames)
        
        # Exclude fields
        if options.exclude_fields:
            fieldnames = [f for f in fieldnames if f not in options.exclude_fields]
        
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter=options.csv_delimiter,
            extrasaction="ignore"
        )
        
        writer.writeheader()
        
        for item in data:
            if isinstance(item, dict):
                # Flatten nested values
                row = {}
                for key in fieldnames:
                    value = item.get(key, "")
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row[key] = value
                writer.writerow(row)
        
        return output.getvalue().encode("utf-8")


class MarkdownExporter:
    """Export to Markdown format."""
    
    def export(
        self,
        data: Any,
        options: ExportOptions,
        title: str = "Export"
    ) -> bytes:
        """Export data to Markdown bytes."""
        lines = []
        
        # Title
        lines.append(f"# {title}")
        lines.append("")
        
        # Metadata
        if options.include_metadata:
            lines.append(f"*Exported: {datetime.now().strftime(options.timestamp_format)}*")
            lines.append("")
        
        # Content
        if isinstance(data, list):
            for i, item in enumerate(data, 1):
                lines.append(f"## Item {i}")
                lines.append("")
                lines.extend(self._format_item(item, options))
                lines.append("")
        elif isinstance(data, dict):
            lines.extend(self._format_item(data, options))
        else:
            lines.append(str(data))
        
        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by Enigma AI Engine*")
        
        return "\n".join(lines).encode("utf-8")
    
    def _format_item(self, item: dict, options: ExportOptions) -> list[str]:
        """Format a single item."""
        lines = []
        
        if isinstance(item, dict):
            fields = options.fields or list(item.keys())
            
            if options.exclude_fields:
                fields = [f for f in fields if f not in options.exclude_fields]
            
            for key in fields:
                if key in item:
                    value = item[key]
                    if isinstance(value, (dict, list)):
                        lines.append(f"**{key}:**")
                        lines.append("```json")
                        lines.append(json.dumps(value, indent=2))
                        lines.append("```")
                    else:
                        lines.append(f"**{key}:** {value}")
        else:
            lines.append(str(item))
        
        return lines


class HTMLExporter:
    """Export to HTML format."""
    
    CSS = """
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; }
        .metadata { color: #888; font-size: 0.9em; margin-bottom: 20px; }
        .item { background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .field { margin: 5px 0; }
        .field-name { font-weight: bold; color: #333; }
        .field-value { color: #666; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f2f2f2; }
    </style>
    """
    
    def export(
        self,
        data: Any,
        options: ExportOptions,
        title: str = "Export"
    ) -> bytes:
        """Export data to HTML bytes."""
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{html.escape(title)}</title>",
            '<meta charset="UTF-8">',
        ]
        
        if options.html_css:
            lines.append(self.CSS)
        
        lines.extend([
            "</head>",
            "<body>",
            f"<h1>{html.escape(title)}</h1>",
        ])
        
        if options.include_metadata:
            lines.append(f'<p class="metadata">Exported: {datetime.now().strftime(options.timestamp_format)}</p>')
        
        # Content
        if isinstance(data, list) and data:
            if all(isinstance(item, dict) for item in data):
                # Table format for list of dicts
                lines.extend(self._format_table(data, options))
            else:
                # Individual items
                for i, item in enumerate(data, 1):
                    lines.append(f'<div class="item">')
                    lines.append(f"<h2>Item {i}</h2>")
                    lines.extend(self._format_item(item, options))
                    lines.append("</div>")
        elif isinstance(data, dict):
            lines.append('<div class="item">')
            lines.extend(self._format_item(data, options))
            lines.append("</div>")
        else:
            lines.append(f"<p>{html.escape(str(data))}</p>")
        
        lines.extend([
            "<hr>",
            "<p><em>Generated by Enigma AI Engine</em></p>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(lines).encode("utf-8")
    
    def _format_table(self, items: list[dict], options: ExportOptions) -> list[str]:
        """Format list of dicts as table."""
        lines = ["<table>"]
        
        # Get fields
        fields = options.fields or sorted({k for item in items for k in item.keys()})
        if options.exclude_fields:
            fields = [f for f in fields if f not in options.exclude_fields]
        
        # Header
        lines.append("<tr>")
        for field in fields:
            lines.append(f"<th>{html.escape(str(field))}</th>")
        lines.append("</tr>")
        
        # Rows
        for item in items:
            lines.append("<tr>")
            for field in fields:
                value = item.get(field, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                lines.append(f"<td>{html.escape(str(value))}</td>")
            lines.append("</tr>")
        
        lines.append("</table>")
        return lines
    
    def _format_item(self, item: dict, options: ExportOptions) -> list[str]:
        """Format a single item."""
        lines = []
        
        if isinstance(item, dict):
            fields = options.fields or list(item.keys())
            if options.exclude_fields:
                fields = [f for f in fields if f not in options.exclude_fields]
            
            for key in fields:
                if key in item:
                    value = item[key]
                    lines.append('<div class="field">')
                    lines.append(f'<span class="field-name">{html.escape(str(key))}:</span> ')
                    
                    if isinstance(value, (dict, list)):
                        lines.append(f'<pre>{html.escape(json.dumps(value, indent=2))}</pre>')
                    else:
                        lines.append(f'<span class="field-value">{html.escape(str(value))}</span>')
                    
                    lines.append("</div>")
        else:
            lines.append(f"<p>{html.escape(str(item))}</p>")
        
        return lines


class TextExporter:
    """Export to plain text format."""
    
    def export(
        self,
        data: Any,
        options: ExportOptions,
        title: str = "Export"
    ) -> bytes:
        """Export data to text bytes."""
        lines = []
        
        # Title
        lines.append("=" * 60)
        lines.append(title.upper())
        lines.append("=" * 60)
        lines.append("")
        
        if options.include_metadata:
            lines.append(f"Exported: {datetime.now().strftime(options.timestamp_format)}")
            lines.append("")
        
        # Content
        if isinstance(data, list):
            for i, item in enumerate(data, 1):
                lines.append("-" * 40)
                lines.append(f"ITEM {i}")
                lines.append("-" * 40)
                lines.extend(self._format_item(item, options))
                lines.append("")
        elif isinstance(data, dict):
            lines.extend(self._format_item(data, options))
        else:
            lines.append(str(data))
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("Generated by Enigma AI Engine")
        
        return "\n".join(lines).encode("utf-8")
    
    def _format_item(self, item: Any, options: ExportOptions) -> list[str]:
        """Format item to text lines."""
        lines = []
        
        if isinstance(item, dict):
            fields = options.fields or list(item.keys())
            if options.exclude_fields:
                fields = [f for f in fields if f not in options.exclude_fields]
            
            max_key_len = max(len(str(k)) for k in fields) if fields else 0
            
            for key in fields:
                if key in item:
                    value = item[key]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    lines.append(f"{str(key).ljust(max_key_len)}: {value}")
        else:
            lines.append(str(item))
        
        return lines


# =============================================================================
# MAIN EXPORTER
# =============================================================================

class DataExporter:
    """
    Universal data exporter.
    
    Features:
    - Multiple formats
    - Compression
    - File or bytes output
    - Batch exports
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Default output directory
        """
        self.output_dir = output_dir or Path("outputs/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._exporters = {
            ExportFormat.JSON: JSONExporter(),
            ExportFormat.JSON_PRETTY: JSONExporter(),
            ExportFormat.CSV: CSVExporter(),
            ExportFormat.MARKDOWN: MarkdownExporter(),
            ExportFormat.HTML: HTMLExporter(),
            ExportFormat.TEXT: TextExporter()
        }
    
    def export(
        self,
        data: Any,
        format: ExportFormat = ExportFormat.JSON,
        options: Optional[ExportOptions] = None,
        title: str = "Export",
        filename: Optional[str] = None,
        save_to_file: bool = True
    ) -> ExportResult:
        """
        Export data.
        
        Args:
            data: Data to export
            format: Export format
            options: Export options
            title: Export title
            filename: Output filename (auto-generated if None)
            save_to_file: Save to file or return bytes
            
        Returns:
            Export result
        """
        start = datetime.now()
        options = options or ExportOptions(format=format)
        options.format = format
        
        try:
            # Get exporter
            exporter = self._exporters.get(format)
            if not exporter:
                return ExportResult(
                    success=False,
                    format=format,
                    size_bytes=0,
                    item_count=0,
                    error=f"Unsupported format: {format.value}"
                )
            
            # Export
            if format in [ExportFormat.MARKDOWN, ExportFormat.HTML, ExportFormat.TEXT]:
                export_data = exporter.export(data, options, title)
            else:
                export_data = exporter.export(data, options)
            
            # Compress
            export_data = self._compress(export_data, options.compression)
            
            # Calculate item count
            item_count = len(data) if isinstance(data, list) else 1
            
            duration = (datetime.now() - start).total_seconds() * 1000
            
            result = ExportResult(
                success=True,
                format=format,
                size_bytes=len(export_data),
                item_count=item_count,
                data=export_data,
                duration_ms=duration
            )
            
            # Save to file
            if save_to_file:
                file_path = self._get_file_path(
                    filename, title, format, options
                )
                
                with open(file_path, "wb") as f:
                    f.write(export_data)
                
                result.file_path = file_path
            
            return result
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                format=format,
                size_bytes=0,
                item_count=0,
                error=str(e),
                duration_ms=(datetime.now() - start).total_seconds() * 1000
            )
    
    def export_batch(
        self,
        items: dict[str, Any],
        format: ExportFormat = ExportFormat.JSON,
        options: Optional[ExportOptions] = None
    ) -> dict[str, ExportResult]:
        """
        Export multiple datasets.
        
        Args:
            items: Dict of name -> data
            format: Export format
            options: Export options
            
        Returns:
            Dict of name -> result
        """
        results = {}
        
        for name, data in items.items():
            results[name] = self.export(
                data=data,
                format=format,
                options=options,
                title=name
            )
        
        return results
    
    def _compress(
        self,
        data: bytes,
        compression: CompressionType
    ) -> bytes:
        """Apply compression."""
        if compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.ZIP:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("data.txt", data)
            return buffer.getvalue()
        return data
    
    def _get_file_path(
        self,
        filename: Optional[str],
        title: str,
        format: ExportFormat,
        options: ExportOptions
    ) -> Path:
        """Generate file path."""
        if filename:
            return self.output_dir / filename
        
        # Generate from template
        ext = format.value.split("_")[0]  # Handle json_pretty -> json
        
        if options.compression == CompressionType.GZIP:
            ext += ".gz"
        elif options.compression == CompressionType.ZIP:
            ext = "zip"
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^\w\-]", "_", title)
        
        filename = options.filename_template.format(
            name=safe_title,
            date=date_str,
            ext=ext
        )
        
        return self.output_dir / filename


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_exporter: Optional[DataExporter] = None


def get_exporter() -> DataExporter:
    """Get or create data exporter."""
    global _exporter
    if _exporter is None:
        _exporter = DataExporter()
    return _exporter


def export_to_json(data: Any, filename: Optional[str] = None, pretty: bool = True) -> ExportResult:
    """Quick export to JSON."""
    format = ExportFormat.JSON_PRETTY if pretty else ExportFormat.JSON
    return get_exporter().export(data, format=format, filename=filename)


def export_to_csv(data: list[dict], filename: Optional[str] = None) -> ExportResult:
    """Quick export to CSV."""
    return get_exporter().export(data, format=ExportFormat.CSV, filename=filename)


def export_to_markdown(data: Any, title: str = "Export", filename: Optional[str] = None) -> ExportResult:
    """Quick export to Markdown."""
    return get_exporter().export(data, format=ExportFormat.MARKDOWN, title=title, filename=filename)


def export_to_html(data: Any, title: str = "Export", filename: Optional[str] = None) -> ExportResult:
    """Quick export to HTML."""
    return get_exporter().export(data, format=ExportFormat.HTML, title=title, filename=filename)
