"""
Tool Usage Analytics
====================

Track and analyze tool usage patterns for insights and optimization.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Record of a single tool usage."""
    tool_name: str
    timestamp: datetime
    success: bool
    duration_ms: float
    params_hash: str  # Hash of parameters for grouping


class ToolAnalytics:
    """
    Track and analyze tool usage patterns.
    
    Features:
    - Usage frequency tracking
    - Success rate monitoring
    - Performance analytics
    - Peak usage hours detection
    - Trend analysis
    """
    
    def __init__(self, max_records: int = 10000):
        """
        Initialize tool analytics.
        
        Args:
            max_records: Maximum records to keep in memory
        """
        self.max_records = max_records
        self.records: list[UsageRecord] = []
        
        logger.info(f"ToolAnalytics initialized (max_records={max_records})")
    
    def record_usage(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        params_hash: str = ""
    ):
        """
        Record a tool usage.
        
        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
            duration_ms: Execution duration in milliseconds
            params_hash: Hash of parameters (for grouping similar calls)
        """
        record = UsageRecord(
            tool_name=tool_name,
            timestamp=datetime.now(),
            success=success,
            duration_ms=duration_ms,
            params_hash=params_hash
        )
        
        self.records.append(record)
        
        # Maintain max size (circular buffer)
        if len(self.records) > self.max_records:
            self.records.pop(0)
    
    def get_usage_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tool_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Generate usage report.
        
        Args:
            start_time: Start of time range (None for no lower bound)
            end_time: End of time range (None for no upper bound)
            tool_name: Filter by tool name (None for all tools)
            
        Returns:
            Dictionary with usage statistics
        """
        # Filter records
        filtered = self._filter_records(start_time, end_time, tool_name)
        
        if not filtered:
            return {
                "total_calls": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "tools_used": [],
            }
        
        # Calculate statistics
        total_calls = len(filtered)
        success_count = sum(1 for r in filtered if r.success)
        success_rate = (success_count / total_calls * 100) if total_calls > 0 else 0
        
        # Average duration
        avg_duration = sum(r.duration_ms for r in filtered) / total_calls
        
        # Per-tool stats
        tool_stats = self._calculate_tool_stats(filtered)
        
        # Peak hours
        peak_hours = self._calculate_peak_hours(filtered)
        
        # Most used tools
        tool_counts = defaultdict(int)
        for r in filtered:
            tool_counts[r.tool_name] += 1
        
        most_used = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Slowest tools
        tool_durations = defaultdict(list)
        for r in filtered:
            tool_durations[r.tool_name].append(r.duration_ms)
        
        slowest = []
        for tool, durations in tool_durations.items():
            avg = sum(durations) / len(durations)
            slowest.append((tool, avg))
        slowest.sort(key=lambda x: x[1], reverse=True)
        slowest = slowest[:10]
        
        return {
            "total_calls": total_calls,
            "success_count": success_count,
            "failure_count": total_calls - success_count,
            "success_rate": round(success_rate, 2),
            "avg_duration_ms": round(avg_duration, 2),
            "tools_used": list(tool_counts.keys()),
            "most_used_tools": [{"tool": t, "count": c} for t, c in most_used],
            "slowest_tools": [{"tool": t, "avg_duration_ms": round(d, 2)} for t, d in slowest],
            "peak_hours": peak_hours,
            "per_tool_stats": tool_stats,
        }
    
    def _filter_records(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        tool_name: Optional[str]
    ) -> list[UsageRecord]:
        """Filter records by criteria."""
        filtered = []
        
        for record in self.records:
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            if tool_name and record.tool_name != tool_name:
                continue
            
            filtered.append(record)
        
        return filtered
    
    def _calculate_tool_stats(self, records: list[UsageRecord]) -> dict[str, Any]:
        """Calculate per-tool statistics."""
        tool_data = defaultdict(lambda: {
            "count": 0,
            "success": 0,
            "durations": []
        })
        
        for record in records:
            data = tool_data[record.tool_name]
            data["count"] += 1
            if record.success:
                data["success"] += 1
            data["durations"].append(record.duration_ms)
        
        # Calculate final stats
        stats = {}
        for tool, data in tool_data.items():
            count = data["count"]
            success = data["success"]
            durations = data["durations"]
            
            stats[tool] = {
                "total_calls": count,
                "success_count": success,
                "failure_count": count - success,
                "success_rate": round((success / count * 100) if count > 0 else 0, 2),
                "avg_duration_ms": round(sum(durations) / len(durations), 2),
                "min_duration_ms": round(min(durations), 2),
                "max_duration_ms": round(max(durations), 2),
            }
        
        return stats
    
    def _calculate_peak_hours(self, records: list[UsageRecord]) -> list[dict[str, Any]]:
        """Calculate peak usage hours."""
        hour_counts = defaultdict(int)
        
        for record in records:
            hour = record.timestamp.hour
            hour_counts[hour] += 1
        
        # Sort by count
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"hour": hour, "count": count}
            for hour, count in sorted_hours[:5]
        ]
    
    def get_trends(self, period: timedelta = timedelta(days=7)) -> dict[str, Any]:
        """
        Analyze usage trends over a period.
        
        Args:
            period: Time period to analyze
            
        Returns:
            Dictionary with trend data
        """
        cutoff_time = datetime.now() - period
        filtered = self._filter_records(start_time=cutoff_time, end_time=None, tool_name=None)
        
        if not filtered:
            return {
                "period_days": period.days,
                "total_calls": 0,
                "daily_average": 0.0,
                "trend": "no data",
            }
        
        # Calculate daily usage
        daily_counts = defaultdict(int)
        for record in filtered:
            date_key = record.timestamp.date()
            daily_counts[date_key] += 1
        
        # Calculate trend
        sorted_dates = sorted(daily_counts.keys())
        if len(sorted_dates) > 1:
            first_half_avg = sum(daily_counts[d] for d in sorted_dates[:len(sorted_dates)//2]) / (len(sorted_dates)//2)
            second_half_avg = sum(daily_counts[d] for d in sorted_dates[len(sorted_dates)//2:]) / (len(sorted_dates) - len(sorted_dates)//2)
            
            if second_half_avg > first_half_avg * 1.1:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"
        
        return {
            "period_days": period.days,
            "total_calls": len(filtered),
            "daily_average": round(len(filtered) / max(period.days, 1), 2),
            "trend": trend,
            "daily_counts": {str(date): count for date, count in daily_counts.items()},
        }
    
    def get_tool_comparison(self, tool_names: list[str]) -> dict[str, Any]:
        """
        Compare usage statistics between tools.
        
        Args:
            tool_names: List of tool names to compare
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        
        for tool_name in tool_names:
            filtered = self._filter_records(None, None, tool_name)
            
            if filtered:
                total = len(filtered)
                success = sum(1 for r in filtered if r.success)
                avg_duration = sum(r.duration_ms for r in filtered) / total
                
                comparison[tool_name] = {
                    "total_calls": total,
                    "success_rate": round((success / total * 100) if total > 0 else 0, 2),
                    "avg_duration_ms": round(avg_duration, 2),
                }
            else:
                comparison[tool_name] = {
                    "total_calls": 0,
                    "success_rate": 0.0,
                    "avg_duration_ms": 0.0,
                }
        
        return comparison
    
    def clear(self):
        """Clear all analytics data."""
        self.records.clear()
        logger.info("Cleared all analytics data")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get analytics statistics."""
        return {
            "total_records": len(self.records),
            "max_records": self.max_records,
            "oldest_record": self.records[0].timestamp.isoformat() if self.records else None,
            "newest_record": self.records[-1].timestamp.isoformat() if self.records else None,
        }


__all__ = [
    "ToolAnalytics",
    "UsageRecord",
]
