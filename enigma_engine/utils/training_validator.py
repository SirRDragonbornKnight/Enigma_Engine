"""
Training Data Validator - Validate training data before training

Checks training data format, quality, and provides suggestions for improvement.
"""

from pathlib import Path
from typing import Any


class TrainingDataValidator:
    """
    Validates training data for quality and format.
    
    Checks:
    - Format correctness
    - Data quantity
    - Conversation structure
    - Balance and diversity
    - Common issues
    """
    
    # Supported formats
    SUPPORTED_FORMATS = {
        'qa': ['Q:', 'A:', 'User:', 'AI:', 'Human:', 'Assistant:'],
        'conversation': ['User:', 'AI:', 'Human:', 'Assistant:', 'Q:', 'A:'],
        'instruction': ['Instruction:', 'Response:', 'Input:', 'Output:'],
    }
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.suggestions = []
    
    def validate_file(self, file_path: str) -> dict[str, Any]:
        """
        Validate a training data file.
        
        Args:
            file_path: Path to the training data file
            
        Returns:
            Validation report dict
        """
        self.issues = []
        self.warnings = []
        self.suggestions = []
        
        try:
            path = Path(file_path)
            if not path.exists():
                return {
                    'valid': False,
                    'error': 'File not found',
                    'issues': ['File does not exist']
                }
            
            # Read file
            with open(path, encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Run validations
            format_valid = self._check_format(content)
            quantity_ok = self._check_quantity(content)
            structure_ok = self._check_structure(content)
            balance_ok = self._check_balance(content)
            quality_ok = self._check_quality(content)
            
            # Calculate statistics
            stats = self._calculate_statistics(content)
            
            # Determine overall validity
            is_valid = format_valid and quantity_ok and len(self.issues) == 0
            
            return {
                'valid': is_valid,
                'issues': self.issues,
                'warnings': self.warnings,
                'suggestions': self.suggestions,
                'statistics': stats,
                'file_path': str(path),
                'file_size_kb': path.stat().st_size / 1024
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'issues': [f'Failed to validate: {e}']
            }
    
    def _check_format(self, content: str) -> bool:
        """Check if the format is recognized."""
        # Check for common markers
        has_markers = False
        detected_format = None
        
        for format_name, markers in self.SUPPORTED_FORMATS.items():
            for marker in markers:
                if marker in content:
                    has_markers = True
                    detected_format = format_name
                    break
            if has_markers:
                break
        
        if not has_markers:
            self.issues.append(
                "No recognized format markers found (Q:/A:, User:/AI:, etc.)"
            )
            self.suggestions.append(
                "Use format like:\nQ: Your question here?\nA: Your answer here.\n"
            )
            return False
        
        self.suggestions.append(f"Detected format: {detected_format}")
        return True
    
    def _check_quantity(self, content: str) -> bool:
        """Check if there's enough data."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        num_lines = len(lines)
        
        # Estimate number of Q&A pairs
        qa_pairs = 0
        for marker in ['Q:', 'User:', 'Human:', 'Instruction:']:
            qa_pairs += content.count(marker)
        
        if qa_pairs < 10:
            self.issues.append(
                f"Very few training examples found (~{qa_pairs}). Need at least 50 for basic training."
            )
            return False
        elif qa_pairs < 50:
            self.warnings.append(
                f"Limited training examples (~{qa_pairs}). 100+ recommended for good results."
            )
        elif qa_pairs < 200:
            self.warnings.append(
                f"Modest dataset (~{qa_pairs} pairs). 500+ recommended for better quality."
            )
        else:
            self.suggestions.append(
                f"Good dataset size (~{qa_pairs} examples)"
            )
        
        return True
    
    def _check_structure(self, content: str) -> bool:
        """Check conversation structure."""
        lines = content.split('\n')
        
        # Check for empty lines (separators)
        has_separators = any(not line.strip() for line in lines)
        if not has_separators:
            self.warnings.append(
                "No empty lines between conversations. Consider adding blank lines to separate Q&A pairs."
            )
        
        # Check for consistent pairing
        user_count = sum(1 for l in lines if l.startswith(('Q:', 'User:', 'Human:')))
        ai_count = sum(1 for l in lines if l.startswith(('A:', 'AI:', 'Assistant:')))
        
        if abs(user_count - ai_count) > user_count * 0.1:
            self.warnings.append(
                f"Unbalanced pairs: {user_count} prompts, {ai_count} responses. Should be roughly equal."
            )
        
        return True
    
    def _check_balance(self, content: str) -> bool:
        """Check data balance and diversity."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        # Check line length distribution
        lengths = [len(l) for l in lines]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            
            # Check for very short responses
            very_short = sum(1 for l in lengths if l < 20)
            if very_short > len(lengths) * 0.3:
                self.warnings.append(
                    f"{very_short} very short lines found. More detailed responses improve quality."
                )
            
            # Check for very long lines
            very_long = sum(1 for l in lengths if l > 500)
            if very_long > 10:
                self.warnings.append(
                    f"{very_long} very long lines. Consider breaking into smaller chunks."
                )
        
        # Check for repeated content
        line_set = set(lines)
        duplicates = len(lines) - len(line_set)
        if duplicates > len(lines) * 0.1:
            self.warnings.append(
                f"{duplicates} duplicate lines found. Remove duplicates for better training."
            )
        
        return True
    
    def _check_quality(self, content: str) -> bool:
        """Check content quality."""
        # Check encoding issues
        if '�' in content or '\x00' in content:
            self.issues.append(
                "Encoding issues detected. Some characters may not display correctly."
            )
        
        # Check for code/special characters balance
        special_char_ratio = len([c for c in content if not c.isalnum() and not c.isspace()]) / len(content)
        if special_char_ratio > 0.3:
            self.warnings.append(
                "High ratio of special characters. This might be code-heavy or contain formatting issues."
            )
        
        # Check for extremely repetitive patterns
        words = content.lower().split()
        if words:
            unique_words = len(set(words))
            word_diversity = unique_words / len(words)
            if word_diversity < 0.1:
                self.warnings.append(
                    "Low vocabulary diversity. Add more varied examples."
                )
        
        return True
    
    def _calculate_statistics(self, content: str) -> dict[str, Any]:
        """Calculate statistics about the training data."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        # Count examples
        user_lines = [l for l in lines if any(l.startswith(m) for m in ['Q:', 'User:', 'Human:'])]
        ai_lines = [l for l in lines if any(l.startswith(m) for m in ['A:', 'AI:', 'Assistant:'])]
        
        # Calculate lengths
        user_lengths = [len(l) for l in user_lines]
        ai_lengths = [len(l) for l in ai_lines]
        
        return {
            'total_lines': len(lines),
            'total_chars': len(content),
            'estimated_pairs': len(user_lines),
            'user_prompts': len(user_lines),
            'ai_responses': len(ai_lines),
            'avg_prompt_length': sum(user_lengths) / len(user_lengths) if user_lengths else 0,
            'avg_response_length': sum(ai_lengths) / len(ai_lengths) if ai_lengths else 0,
            'unique_lines': len(set(lines)),
            'duplicate_rate': (len(lines) - len(set(lines))) / len(lines) if lines else 0
        }
    
    def validate_text(self, text: str) -> dict[str, Any]:
        """
        Validate training data from text content.
        
        Args:
            text: Training data as string
            
        Returns:
            Validation report dict
        """
        self.issues = []
        self.warnings = []
        self.suggestions = []
        
        # Run validations
        format_valid = self._check_format(text)
        quantity_ok = self._check_quantity(text)
        structure_ok = self._check_structure(text)
        balance_ok = self._check_balance(text)
        quality_ok = self._check_quality(text)
        
        # Calculate statistics
        stats = self._calculate_statistics(text)
        
        # Determine overall validity
        is_valid = format_valid and quantity_ok and len(self.issues) == 0
        
        return {
            'valid': is_valid,
            'issues': self.issues,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'statistics': stats
        }
    
    def generate_report(self, validation_result: dict[str, Any]) -> str:
        """
        Generate a human-readable report.
        
        Args:
            validation_result: Result from validate_file or validate_text
            
        Returns:
            Formatted report string
        """
        lines = ["Training Data Validation Report", "=" * 60, ""]
        
        # Status
        if validation_result['valid']:
            lines.append("[OK] Status: VALID - Ready for training")
        else:
            lines.append("[FAIL] Status: INVALID - Issues must be fixed")
        
        lines.append("")
        
        # Statistics
        if 'statistics' in validation_result:
            stats = validation_result['statistics']
            lines.append("Statistics:")
            lines.append("-" * 60)
            lines.append(f"  Total lines: {stats.get('total_lines', 0)}")
            lines.append(f"  Estimated Q&A pairs: {stats.get('estimated_pairs', 0)}")
            lines.append(f"  Average prompt length: {stats.get('avg_prompt_length', 0):.0f} chars")
            lines.append(f"  Average response length: {stats.get('avg_response_length', 0):.0f} chars")
            lines.append(f"  Duplicate rate: {stats.get('duplicate_rate', 0) * 100:.1f}%")
            lines.append("")
        
        # Issues
        if validation_result.get('issues'):
            lines.append("Issues (must fix):")
            lines.append("-" * 60)
            for issue in validation_result['issues']:
                lines.append(f"  [X] {issue}")
            lines.append("")
        
        # Warnings
        if validation_result.get('warnings'):
            lines.append("Warnings (recommended to fix):")
            lines.append("-" * 60)
            for warning in validation_result['warnings']:
                lines.append(f"  ! {warning}")
            lines.append("")
        
        # Suggestions
        if validation_result.get('suggestions'):
            lines.append("Suggestions:")
            lines.append("-" * 60)
            for suggestion in validation_result['suggestions']:
                lines.append(f"  -> {suggestion}")
            lines.append("")
        
        return "\n".join(lines)


class TrainingDataFormatter:
    """Helper to format training data correctly."""
    
    @staticmethod
    def format_qa_pair(question: str, answer: str) -> str:
        """Format a Q&A pair."""
        return f"Q: {question.strip()}\nA: {answer.strip()}\n"
    
    @staticmethod
    def format_conversation(user_msg: str, ai_msg: str) -> str:
        """Format a conversation turn."""
        return f"User: {user_msg.strip()}\nAI: {ai_msg.strip()}\n"
    
    @staticmethod
    def convert_to_format(content: str, target_format: str = "qa") -> str:
        """
        Convert training data to a specific format.
        
        Args:
            content: Original content
            target_format: Target format ('qa', 'conversation')
            
        Returns:
            Formatted content
        """
        lines = content.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted.append('')
                continue
            
            # Detect current format and convert
            if line.startswith('Q:'):
                if target_format == 'conversation':
                    formatted.append('User:' + line[2:])
                else:
                    formatted.append(line)
            elif line.startswith('A:'):
                if target_format == 'conversation':
                    formatted.append('AI:' + line[2:])
                else:
                    formatted.append(line)
            elif line.startswith('User:'):
                if target_format == 'qa':
                    formatted.append('Q:' + line[5:])
                else:
                    formatted.append(line)
            elif line.startswith('AI:') or line.startswith('Assistant:'):
                if target_format == 'qa':
                    formatted.append('A:' + line.split(':', 1)[1])
                else:
                    formatted.append('AI:' + line.split(':', 1)[1])
            else:
                formatted.append(line)
        
        return '\n'.join(formatted)


if __name__ == "__main__":
    # Test validator
    validator = TrainingDataValidator()
    
    # Test with sample data
    sample_data = """
Q: What is Python?
A: Python is a high-level programming language known for its simplicity and readability.

Q: How do I install Python?
A: You can download Python from python.org and run the installer for your operating system.

Q: What are Python's main features?
A: Python has dynamic typing, automatic memory management, extensive libraries, and clear syntax.
"""
    
    result = validator.validate_text(sample_data)
    print(validator.generate_report(result))
    
    # Test formatter
    print("\n\nFormatter Test:")
    print("=" * 60)
    formatted = TrainingDataFormatter.format_qa_pair(
        "What is AI?",
        "AI stands for Artificial Intelligence, the simulation of human intelligence by machines."
    )
    print(formatted)
