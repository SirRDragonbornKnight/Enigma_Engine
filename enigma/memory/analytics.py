"""
Memory Analytics for Enigma Engine
Provides usage statistics and access pattern analysis.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import Counter

from .categorization import Memory, MemoryType, MemoryCategorization

logger = logging.getLogger(__name__)


class MemoryAnalytics:
    """Analyze memory usage patterns."""
    
    def __init__(self, memory_system: MemoryCategorization):
        """
        Initialize memory analytics.
        
        Args:
            memory_system: Memory categorization system
        """
        self.memory_system = memory_system
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with various statistics
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {
                'total_memories': 0,
                'by_type': {},
                'average_importance': 0,
                'total_accesses': 0
            }
        
        # Basic stats
        stats = {
            'total_memories': len(memories),
            'by_type': {},
            'average_importance': sum(m.importance for m in memories) / len(memories),
            'total_accesses': sum(m.access_count for m in memories),
            'average_accesses': sum(m.access_count for m in memories) / len(memories)
        }
        
        # By type
        for mem_type in MemoryType:
            type_memories = self.memory_system.get_memories_by_type(mem_type)
            stats['by_type'][mem_type.value] = {
                'count': len(type_memories),
                'avg_importance': sum(m.importance for m in type_memories) / len(type_memories) if type_memories else 0,
                'avg_accesses': sum(m.access_count for m in type_memories) / len(type_memories) if type_memories else 0
            }
        
        # Age stats
        now = datetime.now().timestamp()
        ages = [(now - m.timestamp) / 86400 for m in memories]  # in days
        
        stats['age'] = {
            'oldest_days': max(ages) if ages else 0,
            'newest_days': min(ages) if ages else 0,
            'average_days': sum(ages) / len(ages) if ages else 0
        }
        
        # Content stats
        stats['content'] = {
            'total_characters': sum(len(m.content) for m in memories),
            'average_length': sum(len(m.content) for m in memories) / len(memories),
            'longest': max(len(m.content) for m in memories),
            'shortest': min(len(m.content) for m in memories)
        }
        
        return stats
    
    def get_access_patterns(self) -> Dict[str, Any]:
        """
        Analyze how memories are accessed.
        
        Returns:
            Access pattern analysis
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {'patterns': {}}
        
        # Sort by access count
        by_access = sorted(memories, key=lambda m: m.access_count, reverse=True)
        
        # Identify patterns
        patterns = {
            'most_accessed': [
                {
                    'id': m.id,
                    'type': m.memory_type.value,
                    'access_count': m.access_count,
                    'content_preview': m.content[:100]
                }
                for m in by_access[:10]
            ],
            'never_accessed': len([m for m in memories if m.access_count == 0]),
            'access_distribution': {}
        }
        
        # Distribution by access count ranges
        access_ranges = [
            ('0', 0, 0),
            ('1-5', 1, 5),
            ('6-10', 6, 10),
            ('11-50', 11, 50),
            ('51+', 51, float('inf'))
        ]
        
        for range_name, min_access, max_access in access_ranges:
            count = len([
                m for m in memories
                if min_access <= m.access_count <= max_access
            ])
            patterns['access_distribution'][range_name] = count
        
        # Recency of access
        now = datetime.now().timestamp()
        patterns['last_access'] = {
            'last_hour': len([m for m in memories if now - m.last_accessed < 3600]),
            'last_day': len([m for m in memories if now - m.last_accessed < 86400]),
            'last_week': len([m for m in memories if now - m.last_accessed < 604800]),
            'older': len([m for m in memories if now - m.last_accessed >= 604800])
        }
        
        return patterns
    
    def get_growth_rate(self, period_days: int = 30) -> float:
        """
        Calculate memory growth rate.
        
        Args:
            period_days: Period to analyze
            
        Returns:
            Average memories created per day
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return 0.0
        
        # Get memories created in the period
        cutoff = datetime.now().timestamp() - (period_days * 86400)
        recent_memories = [m for m in memories if m.timestamp >= cutoff]
        
        # Calculate rate
        return len(recent_memories) / period_days if period_days > 0 else 0.0
    
    def identify_unused_memories(
        self,
        days_threshold: int = 30,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """
        Find memories that haven't been accessed recently.
        
        Args:
            days_threshold: Days since last access
            min_importance: Minimum importance to consider
            
        Returns:
            List of unused memories
        """
        memories = self.memory_system.get_all_memories()
        
        cutoff = datetime.now().timestamp() - (days_threshold * 86400)
        
        unused = [
            m for m in memories
            if m.last_accessed < cutoff and m.importance >= min_importance
        ]
        
        # Sort by last accessed (oldest first)
        unused.sort(key=lambda m: m.last_accessed)
        
        return unused
    
    def get_importance_analysis(self) -> Dict[str, Any]:
        """
        Analyze importance distribution and patterns.
        
        Returns:
            Importance analysis
        """
        memories = self.memory_system.get_all_memories()
        
        if not memories:
            return {'analysis': {}}
        
        # Sort by importance
        by_importance = sorted(memories, key=lambda m: m.importance, reverse=True)
        
        analysis = {
            'most_important': [
                {
                    'id': m.id,
                    'type': m.memory_type.value,
                    'importance': m.importance,
                    'access_count': m.access_count,
                    'content_preview': m.content[:100]
                }
                for m in by_importance[:10]
            ],
            'least_important': [
                {
                    'id': m.id,
                    'type': m.memory_type.value,
                    'importance': m.importance,
                    'content_preview': m.content[:100]
                }
                for m in by_importance[-10:]
            ],
            'distribution': {}
        }
        
        # Distribution by importance ranges
        ranges = [
            ('Very Low (0.0-0.2)', 0.0, 0.2),
            ('Low (0.2-0.4)', 0.2, 0.4),
            ('Medium (0.4-0.6)', 0.4, 0.6),
            ('High (0.6-0.8)', 0.6, 0.8),
            ('Very High (0.8-1.0)', 0.8, 1.0)
        ]
        
        for range_name, min_imp, max_imp in ranges:
            count = len([
                m for m in memories
                if min_imp <= m.importance < max_imp or (max_imp == 1.0 and m.importance == 1.0)
            ])
            analysis['distribution'][range_name] = count
        
        # Correlation with access count
        if len(memories) > 1:
            import numpy as np
            importances = [m.importance for m in memories]
            accesses = [m.access_count for m in memories]
            
            # Simple correlation
            correlation = np.corrcoef(importances, accesses)[0, 1]
            analysis['importance_access_correlation'] = float(correlation)
        
        return analysis
    
    def get_type_analysis(self) -> Dict[str, Any]:
        """
        Analyze memory type usage patterns.
        
        Returns:
            Type analysis
        """
        analysis = {}
        
        for mem_type in MemoryType:
            memories = self.memory_system.get_memories_by_type(mem_type)
            
            if not memories:
                analysis[mem_type.value] = {
                    'count': 0,
                    'percentage': 0.0
                }
                continue
            
            total_memories = len(self.memory_system.get_all_memories())
            
            analysis[mem_type.value] = {
                'count': len(memories),
                'percentage': (len(memories) / total_memories * 100) if total_memories > 0 else 0,
                'avg_importance': sum(m.importance for m in memories) / len(memories),
                'avg_accesses': sum(m.access_count for m in memories) / len(memories),
                'avg_length': sum(len(m.content) for m in memories) / len(memories),
                'total_characters': sum(len(m.content) for m in memories)
            }
        
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate a human-readable analytics report.
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        access_patterns = self.get_access_patterns()
        growth_rate = self.get_growth_rate()
        importance_analysis = self.get_importance_analysis()
        type_analysis = self.get_type_analysis()
        
        report = f"""
ENIGMA ENGINE MEMORY ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}

OVERVIEW
--------
Total Memories: {stats['total_memories']}
Average Importance: {stats['average_importance']:.2f}
Total Accesses: {stats['total_accesses']}
Average Accesses per Memory: {stats['average_accesses']:.2f}

CONTENT STATISTICS
------------------
Total Characters: {stats['content']['total_characters']:,}
Average Length: {stats['content']['average_length']:.1f} characters
Longest Memory: {stats['content']['longest']} characters
Shortest Memory: {stats['content']['shortest']} characters

AGE STATISTICS
--------------
Oldest Memory: {stats['age']['oldest_days']:.1f} days ago
Newest Memory: {stats['age']['newest_days']:.1f} days ago
Average Age: {stats['age']['average_days']:.1f} days

MEMORY TYPES
------------
"""
        
        for mem_type, type_stats in stats['by_type'].items():
            report += f"{mem_type}: {type_stats['count']} memories "
            report += f"(avg importance: {type_stats['avg_importance']:.2f}, "
            report += f"avg accesses: {type_stats['avg_accesses']:.1f})\n"
        
        report += f"""
ACCESS PATTERNS
---------------
Never Accessed: {access_patterns['never_accessed']} memories
Most Accessed Memory: {access_patterns['most_accessed'][0]['access_count'] if access_patterns['most_accessed'] else 0} accesses

Access Distribution:
"""
        
        for range_name, count in access_patterns['access_distribution'].items():
            report += f"  {range_name} accesses: {count} memories\n"
        
        report += f"""
Recent Access:
  Last Hour: {access_patterns['last_access']['last_hour']} memories
  Last Day: {access_patterns['last_access']['last_day']} memories
  Last Week: {access_patterns['last_access']['last_week']} memories

GROWTH RATE
-----------
Average: {growth_rate:.2f} memories per day (last 30 days)

IMPORTANCE ANALYSIS
-------------------
"""
        
        for range_name, count in importance_analysis['distribution'].items():
            report += f"{range_name}: {count} memories\n"
        
        if 'importance_access_correlation' in importance_analysis:
            report += f"\nImportance-Access Correlation: {importance_analysis['importance_access_correlation']:.2f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report
