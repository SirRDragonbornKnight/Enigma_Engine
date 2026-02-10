"""
Bias Detection and Ethics Tools for Enigma AI Engine
Scans datasets for biased patterns and offensive content, provides safe reinforcement.
"""
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BiasDetectionResult:
    """Result of bias detection scan."""
    is_biased: bool
    bias_score: float  # 0.0 to 1.0
    issues_found: list[dict[str, Any]]
    statistics: dict[str, Any]
    recommendations: list[str]


class BiasDetector:
    """Detects biased patterns in text data."""
    
    # Potentially biased terms (can be expanded)
    GENDERED_TERMS = {
        'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'boys', 'male', 'gentleman'],
        'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'girls', 'female', 'lady']
    }
    
    STEREOTYPICAL_ASSOCIATIONS = {
        # Profession stereotypes
        'engineer': ['male'],
        'nurse': ['female'],
        'doctor': ['male'],
        'teacher': ['female'],
        'ceo': ['male'],
        'secretary': ['female'],
        
        # Trait stereotypes
        'emotional': ['female'],
        'logical': ['male'],
        'aggressive': ['male'],
        'nurturing': ['female'],
    }
    
    # Age bias indicators
    AGE_TERMS = ['young', 'old', 'elderly', 'senior', 'millennial', 'boomer']
    
    # Racial/ethnic descriptors (for monitoring, not blocking)
    DEMOGRAPHIC_DESCRIPTORS = [
        'race', 'ethnicity', 'color', 'nationality',
        'black', 'white', 'asian', 'hispanic', 'latino'
    ]
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize bias detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sensitivity = self.config.get('sensitivity', 0.5)  # 0.0 to 1.0
        
    def scan_text(self, text: str) -> BiasDetectionResult:
        """
        Scan a single text for bias.
        
        Args:
            text: Text to scan
            
        Returns:
            BiasDetectionResult
        """
        text_lower = text.lower()
        issues = []
        
        # Check gender balance
        male_count = sum(text_lower.count(term) for term in self.GENDERED_TERMS['male'])
        female_count = sum(text_lower.count(term) for term in self.GENDERED_TERMS['female'])
        total_gendered = male_count + female_count
        
        if total_gendered > 0:
            male_ratio = male_count / total_gendered
            if male_ratio > 0.8 or male_ratio < 0.2:
                issues.append({
                    'type': 'gender_imbalance',
                    'description': f'Gender imbalance detected: {male_ratio:.1%} male terms',
                    'severity': 'medium',
                    'male_count': male_count,
                    'female_count': female_count
                })
        
        # Check for stereotypical associations
        for term, stereotypes in self.STEREOTYPICAL_ASSOCIATIONS.items():
            if term in text_lower:
                # Check if stereotypical gender appears nearby
                for stereotype in stereotypes:
                    pattern = rf'\b{term}\b.{{0,50}}\b{stereotype}\b'
                    if re.search(pattern, text_lower):
                        issues.append({
                            'type': 'stereotypical_association',
                            'description': f'Potential stereotype: "{term}" associated with "{stereotype}"',
                            'severity': 'low',
                            'term': term,
                            'association': stereotype
                        })
        
        # Check for demographic descriptors (just track, not necessarily problematic)
        demographic_mentions = sum(
            text_lower.count(term) for term in self.DEMOGRAPHIC_DESCRIPTORS
        )
        
        # Calculate bias score
        bias_indicators = len(issues)
        bias_score = min(1.0, bias_indicators * 0.2)  # Each issue adds 0.2 to score
        
        # Generate recommendations
        recommendations = []
        if bias_score > 0.3:
            recommendations.append("Review gender representation in the text")
        if any(i['type'] == 'stereotypical_association' for i in issues):
            recommendations.append("Avoid reinforcing stereotypes in descriptions")
        
        statistics = {
            'total_words': len(text.split()),
            'gendered_terms': total_gendered,
            'male_ratio': male_count / total_gendered if total_gendered > 0 else 0,
            'female_ratio': female_count / total_gendered if total_gendered > 0 else 0,
            'demographic_mentions': demographic_mentions,
            'issues_count': len(issues)
        }
        
        return BiasDetectionResult(
            is_biased=bias_score > self.sensitivity,
            bias_score=bias_score,
            issues_found=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def scan_dataset(self, texts: list[str]) -> BiasDetectionResult:
        """
        Scan an entire dataset for bias patterns.
        
        Args:
            texts: List of text samples
            
        Returns:
            Aggregated BiasDetectionResult
        """
        all_issues = []
        total_stats = {
            'total_samples': len(texts),
            'biased_samples': 0,
            'total_gendered_terms': 0,
            'total_male_terms': 0,
            'total_female_terms': 0
        }
        
        for text in texts:
            result = self.scan_text(text)
            if result.is_biased:
                total_stats['biased_samples'] += 1
            all_issues.extend(result.issues_found)
            total_stats['total_gendered_terms'] += result.statistics.get('gendered_terms', 0)
            total_stats['total_male_terms'] += int(
                result.statistics.get('male_ratio', 0) * result.statistics.get('gendered_terms', 0)
            )
            total_stats['total_female_terms'] += int(
                result.statistics.get('female_ratio', 0) * result.statistics.get('gendered_terms', 0)
            )
        
        # Calculate overall bias score
        bias_ratio = total_stats['biased_samples'] / total_stats['total_samples'] if total_stats['total_samples'] > 0 else 0
        bias_score = bias_ratio
        
        # Overall gender balance
        total_gendered = total_stats['total_gendered_terms']
        if total_gendered > 0:
            overall_male_ratio = total_stats['total_male_terms'] / total_gendered
            total_stats['overall_male_ratio'] = overall_male_ratio
            total_stats['overall_female_ratio'] = 1 - overall_male_ratio
        
        recommendations = [
            f"Review {total_stats['biased_samples']} samples flagged for potential bias",
            "Consider balancing gender representation across dataset",
            "Audit for stereotypical associations"
        ]
        
        return BiasDetectionResult(
            is_biased=bias_score > self.sensitivity,
            bias_score=bias_score,
            issues_found=all_issues,
            statistics=total_stats,
            recommendations=recommendations
        )


class OffensiveContentFilter:
    """Filters offensive and harmful content."""
    
    # Basic offensive terms list (can be extended)
    OFFENSIVE_TERMS = {
        # Profanity (mild examples, extend as needed)
        'profanity': ['damn', 'hell', 'crap'],
        
        # Hate speech indicators
        'hate_speech': ['hate', 'stupid', 'idiot', 'moron'],
        
        # Sensitive topics that need careful handling
        'sensitive': ['violence', 'weapon', 'drug', 'suicide', 'self-harm']
    }
    
    def __init__(self, blocklist_path: Optional[Path] = None):
        """
        Initialize offensive content filter.
        
        Args:
            blocklist_path: Path to custom blocklist file
        """
        self.blocklist = set()
        
        # Load built-in terms
        for category, terms in self.OFFENSIVE_TERMS.items():
            self.blocklist.update(terms)
        
        # Load custom blocklist
        if blocklist_path and blocklist_path.exists():
            self._load_blocklist(blocklist_path)
    
    def _load_blocklist(self, path: Path):
        """Load custom blocklist from file."""
        try:
            with open(path) as f:
                for line in f:
                    term = line.strip().lower()
                    if term and not term.startswith('#'):
                        self.blocklist.add(term)
            logger.info(f"Loaded {len(self.blocklist)} terms from blocklist")
        except Exception as e:
            logger.error(f"Failed to load blocklist: {e}")
    
    def scan_text(self, text: str) -> dict[str, Any]:
        """
        Scan text for offensive content.
        
        Args:
            text: Text to scan
            
        Returns:
            Dictionary with scan results
        """
        text_lower = text.lower()
        found_terms = []
        
        for term in self.blocklist:
            if re.search(rf'\b{re.escape(term)}\b', text_lower):
                found_terms.append(term)
        
        is_offensive = len(found_terms) > 0
        
        return {
            'is_offensive': is_offensive,
            'severity': 'high' if len(found_terms) > 2 else 'medium' if found_terms else 'none',
            'found_terms': found_terms,
            'term_count': len(found_terms)
        }
    
    def scan_dataset(self, texts: list[str]) -> dict[str, Any]:
        """
        Scan dataset for offensive content.
        
        Args:
            texts: List of text samples
            
        Returns:
            Aggregated results
        """
        offensive_count = 0
        all_found_terms = []
        
        for text in texts:
            result = self.scan_text(text)
            if result['is_offensive']:
                offensive_count += 1
                all_found_terms.extend(result['found_terms'])
        
        term_frequency = Counter(all_found_terms)
        
        return {
            'total_samples': len(texts),
            'offensive_samples': offensive_count,
            'offensive_ratio': offensive_count / len(texts) if texts else 0,
            'most_common_terms': term_frequency.most_common(10),
            'unique_offensive_terms': len(set(all_found_terms))
        }
    
    def filter_text(self, text: str, replacement: str = "[FILTERED]") -> str:
        """
        Filter offensive terms from text.
        
        Args:
            text: Text to filter
            replacement: Replacement string for offensive terms
            
        Returns:
            Filtered text
        """
        filtered = text
        for term in self.blocklist:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            filtered = pattern.sub(replacement, filtered)
        
        return filtered


class SafeReinforcementLogic:
    """Safe reinforcement logic for inference to prevent harmful outputs."""
    
    def __init__(self):
        """Initialize safe reinforcement system."""
        self.bias_detector = BiasDetector()
        self.content_filter = OffensiveContentFilter()
        
        # Safety rules
        self.safety_rules = [
            "Do not generate content that could cause harm",
            "Avoid reinforcing stereotypes or biases",
            "Be respectful and inclusive",
            "Do not generate offensive or hateful content",
            "Be mindful of sensitive topics"
        ]
    
    def check_output_safety(self, text: str) -> dict[str, Any]:
        """
        Check if AI output is safe and ethical.
        
        Args:
            text: Generated text to check
            
        Returns:
            Safety check results
        """
        # Check for bias
        bias_result = self.bias_detector.scan_text(text)
        
        # Check for offensive content
        offensive_result = self.content_filter.scan_text(text)
        
        # Determine if output is safe
        is_safe = not offensive_result['is_offensive'] and bias_result.bias_score < 0.5
        
        issues = []
        if offensive_result['is_offensive']:
            issues.append({
                'type': 'offensive_content',
                'severity': offensive_result['severity'],
                'details': f"Found {len(offensive_result['found_terms'])} offensive terms"
            })
        
        if bias_result.is_biased:
            issues.append({
                'type': 'potential_bias',
                'severity': 'medium',
                'details': f"Bias score: {bias_result.bias_score:.2f}"
            })
        
        return {
            'is_safe': is_safe,
            'confidence': 1.0 - max(bias_result.bias_score, 0.5 if offensive_result['is_offensive'] else 0),
            'issues': issues,
            'bias_result': bias_result,
            'offensive_result': offensive_result,
            'should_regenerate': not is_safe
        }
    
    def get_safety_prompt_additions(self) -> str:
        """Get safety guidelines to add to system prompt."""
        return "Safety guidelines:\n" + "\n".join(f"- {rule}" for rule in self.safety_rules)


def scan_training_data(
    data_path: Path,
    output_report_path: Optional[Path] = None
) -> dict[str, Any]:
    """
    Scan training data for bias and offensive content.
    
    Args:
        data_path: Path to training data file
        output_report_path: Optional path to save report
        
    Returns:
        Scan report
    """
    logger.info(f"Scanning training data: {data_path}")
    
    # Load data
    texts = []
    try:
        with open(data_path, encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {'error': str(e)}
    
    # Scan for bias
    bias_detector = BiasDetector()
    bias_result = bias_detector.scan_dataset(texts)
    
    # Scan for offensive content
    content_filter = OffensiveContentFilter()
    offensive_result = content_filter.scan_dataset(texts)
    
    # Compile report
    report = {
        'data_path': str(data_path),
        'total_samples': len(texts),
        'bias_scan': {
            'is_biased': bias_result.is_biased,
            'bias_score': bias_result.bias_score,
            'issues_count': len(bias_result.issues_found),
            'recommendations': bias_result.recommendations
        },
        'offensive_content_scan': offensive_result,
        'overall_safety_score': 1.0 - max(bias_result.bias_score, offensive_result['offensive_ratio']),
        'recommendations': bias_result.recommendations
    }
    
    # Save report
    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_report_path}")
    
    return report
