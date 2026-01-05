"""
Memory Visualization for Enigma Engine
Generates visualization data for memory analysis and insights.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

from .categorization import Memory, MemoryType, MemoryCategorization

logger = logging.getLogger(__name__)


class MemoryVisualizer:
    """Generate visualization data for memories."""
    
    def __init__(self, memory_system: MemoryCategorization):
        """
        Initialize memory visualizer.
        
        Args:
            memory_system: Memory categorization system
        """
        self.memory_system = memory_system
    
    def generate_timeline(self, memories: Optional[List[Memory]] = None) -> Dict[str, Any]:
        """
        Generate timeline visualization data.
        
        Args:
            memories: List of memories (None = all memories)
            
        Returns:
            Timeline data suitable for visualization libraries
        """
        if memories is None:
            memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {'events': [], 'range': {}}
        
        # Sort by timestamp
        memories = sorted(memories, key=lambda m: m.timestamp)
        
        # Group by day
        daily_counts = Counter()
        events = []
        
        for memory in memories:
            dt = datetime.fromtimestamp(memory.timestamp)
            day_key = dt.strftime('%Y-%m-%d')
            daily_counts[day_key] += 1
            
            # Add event
            events.append({
                'date': dt.isoformat(),
                'id': memory.id,
                'type': memory.memory_type.value,
                'importance': memory.importance,
                'content_preview': memory.content[:100] if len(memory.content) > 100 else memory.content
            })
        
        # Calculate range
        start_date = datetime.fromtimestamp(memories[0].timestamp)
        end_date = datetime.fromtimestamp(memories[-1].timestamp)
        
        return {
            'events': events,
            'daily_counts': dict(daily_counts),
            'range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days': (end_date - start_date).days + 1
            },
            'total_events': len(events)
        }
    
    def generate_type_distribution(self) -> Dict[str, int]:
        """
        Generate pie chart data for memory types.
        
        Returns:
            Dictionary mapping memory type to count
        """
        distribution = {}
        
        for mem_type in MemoryType:
            count = self.memory_system.categories[mem_type].count(include_expired=False)
            distribution[mem_type.value] = count
        
        return distribution
    
    def generate_access_heatmap(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate heatmap of memory access patterns.
        
        Args:
            days: Number of days to include
            
        Returns:
            Heatmap data with dates and access counts
        """
        memories = self.memory_system.get_all_memories()
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Initialize heatmap data
        heatmap = {}
        current_date = start_date
        
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            heatmap[date_key] = {'accesses': 0, 'unique_memories': set()}
            current_date += timedelta(days=1)
        
        # Count accesses per day
        for memory in memories:
            if memory.last_accessed:
                access_date = datetime.fromtimestamp(memory.last_accessed)
                if start_date <= access_date <= end_date:
                    date_key = access_date.strftime('%Y-%m-%d')
                    if date_key in heatmap:
                        heatmap[date_key]['accesses'] += memory.access_count
                        heatmap[date_key]['unique_memories'].add(memory.id)
        
        # Convert sets to counts
        for date_key in heatmap:
            heatmap[date_key]['unique_memories'] = len(heatmap[date_key]['unique_memories'])
        
        return {
            'heatmap': heatmap,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    
    def generate_importance_histogram(self, bins: int = 10) -> Dict[str, Any]:
        """
        Generate histogram of importance scores.
        
        Args:
            bins: Number of bins for histogram
            
        Returns:
            Histogram data
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {'bins': [], 'counts': []}
        
        # Create bins
        bin_edges = [i / bins for i in range(bins + 1)]
        bin_counts = [0] * bins
        
        # Count memories in each bin
        for memory in memories:
            importance = memory.importance
            
            # Find appropriate bin
            for i in range(bins):
                if bin_edges[i] <= importance < bin_edges[i + 1]:
                    bin_counts[i] += 1
                    break
            else:
                # Handle edge case of importance == 1.0
                if importance == 1.0:
                    bin_counts[-1] += 1
        
        # Format bins
        bin_labels = [
            f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
            for i in range(bins)
        ]
        
        return {
            'bins': bin_labels,
            'counts': bin_counts,
            'total_memories': len(memories)
        }
    
    def generate_memory_growth(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate memory growth over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Growth data
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {'growth': []}
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Count memories created each day
        daily_counts = Counter()
        cumulative_count = 0
        
        for memory in memories:
            mem_date = datetime.fromtimestamp(memory.timestamp)
            if mem_date >= start_date:
                date_key = mem_date.strftime('%Y-%m-%d')
                daily_counts[date_key] += 1
        
        # Generate growth data
        growth_data = []
        current_date = start_date
        
        while current_date <= end_date:
            date_key = current_date.strftime('%Y-%m-%d')
            cumulative_count += daily_counts.get(date_key, 0)
            
            growth_data.append({
                'date': date_key,
                'new_memories': daily_counts.get(date_key, 0),
                'cumulative_total': cumulative_count
            })
            
            current_date += timedelta(days=1)
        
        return {
            'growth': growth_data,
            'total_new': sum(daily_counts.values()),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    
    def export_to_html(self, path: Path, include_charts: bool = True):
        """
        Export interactive visualization to HTML.
        
        Args:
            path: Path to save HTML file
            include_charts: Include chart visualizations (requires Chart.js CDN)
        """
        # Generate all visualizations
        timeline = self.generate_timeline()
        type_dist = self.generate_type_distribution()
        access_heatmap = self.generate_access_heatmap()
        importance_hist = self.generate_importance_histogram()
        growth = self.generate_memory_growth()
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Memory Visualization - Enigma Engine</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .chart-container {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .stat-box {{
            display: inline-block;
            padding: 15px 25px;
            margin: 10px;
            background-color: #007bff;
            color: white;
            border-radius: 4px;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        canvas {{
            max-width: 100%;
        }}
    </style>
    {"<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>" if include_charts else ""}
</head>
<body>
    <div class="container">
        <h1>Memory Visualization Dashboard</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Overview</h2>
        <div class="stat-box">
            <div class="stat-label">Total Memories</div>
            <div class="stat-value">{timeline['total_events']}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Memory Types</div>
            <div class="stat-value">{len([v for v in type_dist.values() if v > 0])}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Days Tracked</div>
            <div class="stat-value">{timeline['range'].get('days', 0)}</div>
        </div>
        
        <h2>Memory Type Distribution</h2>
        <div class="chart-container">
            <pre>{self._format_dict(type_dist)}</pre>
        </div>
        
        <h2>Importance Distribution</h2>
        <div class="chart-container">
            <pre>{self._format_dict(dict(zip(importance_hist['bins'], importance_hist['counts'])))}</pre>
        </div>
        
        <h2>Recent Growth (Last 30 Days)</h2>
        <div class="chart-container">
            <p>Total new memories: {growth['total_new']}</p>
        </div>
        
        <h2>Recent Events</h2>
        <div class="chart-container">
            <ul>
            {''.join([f"<li><strong>{e['type']}</strong> - {e['date']}: {e['content_preview']}</li>" for e in timeline['events'][-10:]])}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding='utf-8')
        
        logger.info(f"Exported visualization to {path}")
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for display."""
        import json
        return json.dumps(d, indent=2)
