"""
Feedback System - Collect user ratings and feedback on AI responses

Allows users to rate responses and provide feedback for continuous improvement.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FeedbackCollector:
    """
    Collects and stores user feedback on AI responses.
    
    Features:
    - Star ratings (1-5)
    - Thumbs up/down
    - Text feedback
    - Feedback categories (helpful, accurate, creative, etc.)
    - Analytics and insights
    """
    
    RATING_CATEGORIES = [
        "helpful",
        "accurate",
        "creative",
        "clear",
        "relevant",
        "complete"
    ]
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize feedback collector.
        
        Args:
            storage_path: Path to store feedback data
        """
        if storage_path is None:
            from ..config import CONFIG
            storage_path = Path(CONFIG["data_dir"]) / "feedback.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.feedback_data = self._load()
    
    def _load(self) -> List[Dict]:
        """Load feedback from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save(self):
        """Save feedback to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def add_rating(self, response_id: str, rating: int, 
                   prompt: str = "", response: str = "",
                   categories: Optional[Dict[str, int]] = None,
                   text_feedback: str = "") -> Dict[str, Any]:
        """
        Add a rating for a response.
        
        Args:
            response_id: Unique ID for the response
            rating: Overall rating (1-5 stars)
            prompt: The user's prompt
            response: The AI's response
            categories: Optional ratings for specific categories
            text_feedback: Optional text feedback
            
        Returns:
            Feedback record
        """
        feedback = {
            'id': response_id,
            'timestamp': datetime.now().isoformat(),
            'rating': rating,
            'prompt': prompt[:500],  # Truncate to save space
            'response': response[:1000],
            'categories': categories or {},
            'text_feedback': text_feedback,
        }
        
        self.feedback_data.append(feedback)
        self._save()
        
        return feedback
    
    def add_thumbs(self, response_id: str, thumbs_up: bool,
                   prompt: str = "", response: str = "",
                   reason: str = "") -> Dict[str, Any]:
        """
        Add thumbs up/down feedback.
        
        Args:
            response_id: Unique ID for the response
            thumbs_up: True for thumbs up, False for thumbs down
            prompt: The user's prompt
            response: The AI's response
            reason: Optional reason for the rating
            
        Returns:
            Feedback record
        """
        feedback = {
            'id': response_id,
            'timestamp': datetime.now().isoformat(),
            'thumbs_up': thumbs_up,
            'prompt': prompt[:500],
            'response': response[:1000],
            'reason': reason,
        }
        
        self.feedback_data.append(feedback)
        self._save()
        
        return feedback
    
    def get_feedback(self, response_id: str) -> Optional[Dict]:
        """Get feedback for a specific response."""
        for feedback in self.feedback_data:
            if feedback.get('id') == response_id:
                return feedback
        return None
    
    def get_all_feedback(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all feedback.
        
        Args:
            limit: Optional limit on number of records to return
            
        Returns:
            List of feedback records
        """
        if limit:
            return self.feedback_data[-limit:]
        return self.feedback_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'average_rating': 0,
                'thumbs_up_percent': 0,
                'category_averages': {}
            }
        
        # Calculate rating statistics
        ratings = [f['rating'] for f in self.feedback_data if 'rating' in f]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Calculate thumbs statistics
        thumbs_data = [f for f in self.feedback_data if 'thumbs_up' in f]
        thumbs_up = sum(1 for f in thumbs_data if f['thumbs_up'])
        thumbs_up_percent = (thumbs_up / len(thumbs_data) * 100) if thumbs_data else 0
        
        # Calculate category averages
        category_sums = {cat: [] for cat in self.RATING_CATEGORIES}
        for feedback in self.feedback_data:
            if 'categories' in feedback:
                for cat, rating in feedback['categories'].items():
                    if cat in category_sums:
                        category_sums[cat].append(rating)
        
        category_averages = {
            cat: (sum(ratings) / len(ratings) if ratings else 0)
            for cat, ratings in category_sums.items()
        }
        
        return {
            'total_feedback': len(self.feedback_data),
            'total_ratings': len(ratings),
            'total_thumbs': len(thumbs_data),
            'average_rating': round(avg_rating, 2),
            'thumbs_up_count': thumbs_up,
            'thumbs_down_count': len(thumbs_data) - thumbs_up,
            'thumbs_up_percent': round(thumbs_up_percent, 1),
            'category_averages': {
                cat: round(avg, 2)
                for cat, avg in category_averages.items()
            }
        }
    
    def get_recent_low_ratings(self, threshold: int = 2, 
                              limit: int = 10) -> List[Dict]:
        """
        Get recent low-rated responses for review.
        
        Args:
            threshold: Rating threshold (responses <= this rating)
            limit: Maximum number to return
            
        Returns:
            List of low-rated feedback
        """
        low_ratings = [
            f for f in self.feedback_data
            if 'rating' in f and f['rating'] <= threshold
        ]
        
        # Sort by timestamp (most recent first)
        low_ratings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return low_ratings[:limit]
    
    def get_high_performing_responses(self, threshold: int = 4,
                                     limit: int = 10) -> List[Dict]:
        """
        Get high-rated responses to learn from.
        
        Args:
            threshold: Rating threshold (responses >= this rating)
            limit: Maximum number to return
            
        Returns:
            List of high-rated feedback
        """
        high_ratings = [
            f for f in self.feedback_data
            if 'rating' in f and f['rating'] >= threshold
        ]
        
        # Sort by timestamp (most recent first)
        high_ratings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return high_ratings[:limit]
    
    def export_for_training(self, output_path: Path,
                           min_rating: int = 4) -> int:
        """
        Export high-quality interactions for training.
        
        Args:
            output_path: Path to export training data
            min_rating: Minimum rating to include
            
        Returns:
            Number of interactions exported
        """
        high_quality = [
            f for f in self.feedback_data
            if 'rating' in f and f['rating'] >= min_rating
            and f.get('prompt') and f.get('response')
        ]
        
        # Format as training data
        training_lines = []
        for feedback in high_quality:
            training_lines.append(f"User: {feedback['prompt']}")
            training_lines.append(f"AI: {feedback['response']}")
            training_lines.append("")  # Blank line separator
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_lines))
        
        return len(high_quality)
    
    def clear_old_feedback(self, days: int = 90):
        """
        Clear feedback older than specified days.
        
        Args:
            days: Number of days to keep
        """
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        self.feedback_data = [
            f for f in self.feedback_data
            if datetime.fromisoformat(f['timestamp']).timestamp() > cutoff
        ]
        
        self._save()


class ResponseRatingWidget:
    """
    Helper class to generate rating UI descriptions.
    
    This can be used by the GUI to create rating buttons.
    """
    
    @staticmethod
    def get_star_rating_html(rating: int) -> str:
        """Generate HTML for star rating display."""
        filled = "★" * rating
        empty = "☆" * (5 - rating)
        return f'<span style="color: gold;">{filled}</span><span style="color: gray;">{empty}</span>'
    
    @staticmethod
    def get_thumbs_html(thumbs_up: bool) -> str:
        """Generate HTML for thumbs display."""
        if thumbs_up:
            return '<span style="color: green; font-size: 20px;">👍</span>'
        else:
            return '<span style="color: red; font-size: 20px;">👎</span>'
    
    @staticmethod
    def get_category_options() -> List[str]:
        """Get list of rating categories."""
        return FeedbackCollector.RATING_CATEGORIES.copy()


if __name__ == "__main__":
    # Test feedback system
    collector = FeedbackCollector()
    
    # Add some test ratings
    collector.add_rating(
        response_id="test_1",
        rating=5,
        prompt="What is Python?",
        response="Python is a high-level programming language...",
        categories={
            "helpful": 5,
            "accurate": 5,
            "clear": 4
        },
        text_feedback="Great explanation!"
    )
    
    collector.add_rating(
        response_id="test_2",
        rating=2,
        prompt="How do I fix this error?",
        response="I'm not sure...",
        categories={
            "helpful": 2,
            "accurate": 1,
            "complete": 1
        },
        text_feedback="Not very helpful"
    )
    
    collector.add_thumbs(
        response_id="test_3",
        thumbs_up=True,
        prompt="Tell me a joke",
        response="Why did the programmer quit? They didn't get arrays!",
        reason="Made me laugh!"
    )
    
    # Print statistics
    print("Feedback Statistics:")
    print("=" * 50)
    stats = collector.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n\nRecent Low Ratings:")
    print("=" * 50)
    for feedback in collector.get_recent_low_ratings():
        print(f"Rating: {feedback['rating']}")
        print(f"Prompt: {feedback['prompt']}")
        print(f"Feedback: {feedback.get('text_feedback', 'N/A')}")
        print()
