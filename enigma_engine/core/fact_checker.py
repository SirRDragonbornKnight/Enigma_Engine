"""
AI Accuracy and Fact-Checking Module

Reduces hallucinations and improves response accuracy through:
1. RAG (Retrieval Augmented Generation) - ground responses in real data
2. Confidence scoring - how sure is the AI about its response
3. "I don't know" responses - avoid making things up
4. Web search verification - check facts against web sources

Usage:
    from enigma_engine.core.fact_checker import FactChecker, ConfidenceScorer
    
    checker = FactChecker()
    
    # Check response accuracy
    result = checker.verify_response(question, response)
    print(f"Confidence: {result.confidence}")
    print(f"Verified facts: {result.verified_facts}")
    print(f"Uncertain claims: {result.uncertain_claims}")
    
    # With RAG
    response = checker.generate_with_rag(question, knowledge_base)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """Result from fact-checking a response."""
    confidence: float  # 0.0 to 1.0
    verified_facts: List[str]  # Facts confirmed as accurate
    uncertain_claims: List[str]  # Claims that couldn't be verified
    contradictions: List[str]  # Claims that contradict known facts
    sources: List[str]  # Sources used for verification
    should_qualify: bool  # Should the response include uncertainty markers
    suggested_response: Optional[str] = None  # Improved response if needed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "verified_facts": self.verified_facts,
            "uncertain_claims": self.uncertain_claims,
            "contradictions": self.contradictions,
            "sources": self.sources,
            "should_qualify": self.should_qualify,
            "suggested_response": self.suggested_response,
        }


@dataclass
class RAGConfig:
    """Configuration for RAG (Retrieval Augmented Generation)."""
    knowledge_paths: List[str] = field(default_factory=list)
    max_context_chunks: int = 5
    similarity_threshold: float = 0.7
    use_web_search: bool = False
    web_search_timeout: float = 10.0
    cache_retrievals: bool = True


class ConfidenceScorer:
    """
    Scores the confidence of AI responses.
    
    Factors considered:
    - Hedging language ("I think", "maybe", "probably")
    - Specificity of claims (vague vs detailed)
    - Citation of sources
    - Internal consistency
    - Match with training data patterns
    """
    
    # Hedging phrases that indicate uncertainty
    UNCERTAINTY_MARKERS = [
        "i think", "i believe", "probably", "maybe", "perhaps",
        "it's possible", "might be", "could be", "likely", "unlikely",
        "i'm not sure", "i'm uncertain", "as far as i know",
        "to my knowledge", "if i recall", "i may be wrong",
        "generally speaking", "in some cases", "it depends",
    ]
    
    # Phrases indicating high confidence
    CONFIDENCE_MARKERS = [
        "definitely", "certainly", "absolutely", "always", "never",
        "without a doubt", "for sure", "100%", "guaranteed",
        "it is", "the fact is", "proven", "established",
    ]
    
    # Factual claim patterns (things that can be verified)
    FACTUAL_PATTERNS = [
        r'\b\d{4}\b',  # Years
        r'\b\d+(?:\.\d+)?%',  # Percentages
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Money
        r'\b\d+(?:\.\d+)?\s*(?:km|miles|meters|feet|kg|pounds|lbs)\b',  # Measurements
        r'(?:according to|as per|based on)\s+[\w\s]+',  # Citations
    ]
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def score(self, response: str, context: Optional[str] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Score the confidence level of a response.
        
        Returns:
            (confidence_score, analysis_dict)
            confidence_score: 0.0 (very uncertain) to 1.0 (very certain)
        """
        response_lower = response.lower()
        analysis = {
            "uncertainty_markers_found": [],
            "confidence_markers_found": [],
            "factual_claims_count": 0,
            "response_length": len(response),
            "has_qualifications": False,
            "vague_count": 0,
        }
        
        # Count uncertainty markers
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in response_lower:
                analysis["uncertainty_markers_found"].append(marker)
        
        # Count confidence markers
        for marker in self.CONFIDENCE_MARKERS:
            if marker in response_lower:
                analysis["confidence_markers_found"].append(marker)
        
        # Count factual claims
        for pattern in self.FACTUAL_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            analysis["factual_claims_count"] += len(matches)
        
        # Check for vague language
        vague_patterns = [
            r'\bsome\b', r'\bseveral\b', r'\bmany\b', r'\bfew\b',
            r'\bsomething\b', r'\bsomewhere\b', r'\bsomeone\b',
            r'\bstuff\b', r'\bthings\b', r'\betc\b',
        ]
        for vp in vague_patterns:
            if re.search(vp, response_lower):
                analysis["vague_count"] += 1
        
        # Calculate base confidence
        uncertainty_penalty = len(analysis["uncertainty_markers_found"]) * 0.1
        confidence_boost = len(analysis["confidence_markers_found"]) * 0.05
        vague_penalty = analysis["vague_count"] * 0.05
        
        # Start with neutral confidence
        confidence = 0.7
        
        # Apply adjustments
        confidence -= uncertainty_penalty
        confidence += confidence_boost
        confidence -= vague_penalty
        
        # Shorter responses with no hedging are suspicious (overconfident)
        if len(response) < 50 and not analysis["uncertainty_markers_found"]:
            confidence -= 0.1
        
        # Responses with citations get a boost
        if analysis["factual_claims_count"] > 0:
            confidence += 0.1
        
        # Clamp to valid range
        confidence = max(0.1, min(0.95, confidence))
        
        # Check if response should include qualifications
        analysis["has_qualifications"] = len(analysis["uncertainty_markers_found"]) > 0
        
        # Store in history
        if len(self.history) >= self._max_history:
            self.history.pop(0)
        self.history.append({
            "response_preview": response[:100],
            "confidence": confidence,
            "analysis": analysis,
        })
        
        return confidence, analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scoring statistics."""
        if not self.history:
            return {"avg_confidence": 0.5, "count": 0}
        
        confidences = [h["confidence"] for h in self.history]
        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "count": len(self.history),
        }


class KnowledgeBase:
    """
    Simple knowledge retrieval for RAG.
    
    Stores facts as searchable chunks for grounding AI responses.
    """
    
    def __init__(self, max_chunks: int = 10000):
        self.chunks: List[Dict[str, Any]] = []
        self.max_chunks = max_chunks
        self._index: Dict[str, List[int]] = {}  # word -> chunk indices
    
    def add_document(self, content: str, source: str = "unknown", chunk_size: int = 500):
        """Add a document, splitting into searchable chunks."""
        # Split into paragraphs first
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            # Split long paragraphs
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = []
                current_len = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_len += len(word) + 1
                    
                    if current_len >= chunk_size:
                        self._add_chunk(' '.join(current_chunk), source)
                        current_chunk = []
                        current_len = 0
                
                if current_chunk:
                    self._add_chunk(' '.join(current_chunk), source)
            else:
                self._add_chunk(para, source)
    
    def _add_chunk(self, text: str, source: str):
        """Add a single chunk and index it."""
        if len(self.chunks) >= self.max_chunks:
            # Remove oldest chunk
            old_chunk = self.chunks.pop(0)
            # Update indices (shift all by -1)
            new_index = {}
            for word, indices in self._index.items():
                new_indices = [i - 1 for i in indices if i > 0]
                if new_indices:
                    new_index[word] = new_indices
            self._index = new_index
        
        chunk_idx = len(self.chunks)
        self.chunks.append({
            "text": text,
            "source": source,
            "index": chunk_idx,
        })
        
        # Index words
        words = set(re.findall(r'\b\w+\b', text.lower()))
        for word in words:
            if word not in self._index:
                self._index[word] = []
            self._index[word].append(chunk_idx)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using keyword matching."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score each chunk by word overlap
        scores: Dict[int, int] = {}
        for word in query_words:
            if word in self._index:
                for idx in self._index[word]:
                    scores[idx] = scores.get(idx, 0) + 1
        
        # Sort by score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for idx, score in sorted_chunks[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = score / len(query_words) if query_words else 0
            results.append(chunk)
        
        return results
    
    def load_from_file(self, path: str):
        """Load knowledge from a text file."""
        path = Path(path)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            self.add_document(content, source=path.name)
            logger.info(f"Loaded {len(self.chunks)} chunks from {path}")
    
    def save_to_file(self, path: str):
        """Save knowledge base to JSON."""
        path = Path(path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.chunks,
                "count": len(self.chunks),
            }, f, indent=2)
    
    def load_from_json(self, path: str):
        """Load knowledge base from JSON."""
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            # Rebuild index
            self._index = {}
            for i, chunk in enumerate(self.chunks):
                words = set(re.findall(r'\b\w+\b', chunk["text"].lower()))
                for word in words:
                    if word not in self._index:
                        self._index[word] = []
                    self._index[word].append(i)


class FactChecker:
    """
    Main fact-checking system for AI responses.
    
    Combines multiple verification methods:
    - RAG context grounding
    - Confidence scoring
    - Self-consistency checking
    - Optional web search verification
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.knowledge = KnowledgeBase()
        self.scorer = ConfidenceScorer()
        
        # Load knowledge from configured paths
        for path in self.config.knowledge_paths:
            self.knowledge.load_from_file(path)
        
        # "I don't know" response templates
        self.idk_templates = [
            "I'm not certain about this. {reason}",
            "I don't have reliable information on {topic}.",
            "This is outside my training data. I'd recommend checking {source}.",
            "I'm not confident I can answer this accurately.",
            "I don't know the answer to this, but I can try to help you find out.",
        ]
    
    def verify_response(
        self,
        question: str,
        response: str,
        use_web: bool = False,
    ) -> FactCheckResult:
        """
        Verify the accuracy of an AI response.
        
        Args:
            question: The question that was asked
            response: The AI's response to verify
            use_web: Whether to use web search for verification
            
        Returns:
            FactCheckResult with confidence and verification details
        """
        # Get confidence score
        confidence, analysis = self.scorer.score(response, context=question)
        
        # Search knowledge base for relevant facts
        relevant_chunks = self.knowledge.search(question, top_k=self.config.max_context_chunks)
        
        verified = []
        uncertain = []
        contradictions = []
        sources = [c["source"] for c in relevant_chunks]
        
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Check each claim against knowledge base
        for claim in claims:
            verification = self._verify_claim(claim, relevant_chunks)
            if verification == "verified":
                verified.append(claim)
            elif verification == "contradicted":
                contradictions.append(claim)
                confidence *= 0.7  # Reduce confidence for contradictions
            else:
                uncertain.append(claim)
        
        # Web search verification (if enabled)
        if use_web and self.config.use_web_search:
            web_results = self._web_verify(claims)
            sources.extend(web_results.get("sources", []))
            # Adjust confidence based on web results
            if web_results.get("verified"):
                confidence = min(0.95, confidence + 0.1)
        
        # Determine if we should qualify the response
        should_qualify = (
            confidence < 0.6 or
            len(contradictions) > 0 or
            len(uncertain) > len(verified)
        )
        
        # Generate improved response if needed
        suggested = None
        if should_qualify:
            suggested = self._generate_qualified_response(
                question, response, confidence, uncertain, contradictions
            )
        
        return FactCheckResult(
            confidence=confidence,
            verified_facts=verified,
            uncertain_claims=uncertain,
            contradictions=contradictions,
            sources=list(set(sources)),
            should_qualify=should_qualify,
            suggested_response=suggested,
        )
    
    def generate_with_rag(
        self,
        question: str,
        model_generate_fn: callable = None,
    ) -> Tuple[str, float]:
        """
        Generate a response grounded in knowledge base.
        
        Args:
            question: The question to answer
            model_generate_fn: Function to generate response, takes (prompt) -> str
            
        Returns:
            (response, confidence_score)
        """
        # Retrieve relevant context
        relevant = self.knowledge.search(question, top_k=self.config.max_context_chunks)
        
        if not relevant:
            # No relevant knowledge - should express uncertainty
            return self._get_idk_response(question), 0.3
        
        # Build context string
        context_parts = []
        for chunk in relevant:
            context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Build RAG prompt
        rag_prompt = f"""Answer the following question using ONLY the information provided in the context below.
If the context doesn't contain enough information to answer confidently, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        if model_generate_fn:
            response = model_generate_fn(rag_prompt)
        else:
            # Placeholder - would integrate with actual model
            response = f"Based on the available information: [Response would be generated here]"
        
        # Score the response
        confidence, _ = self.scorer.score(response, context=context)
        
        return response, confidence
    
    def should_say_idk(self, question: str, confidence: float) -> bool:
        """Determine if the AI should say 'I don't know'."""
        # Check if we have relevant knowledge
        relevant = self.knowledge.search(question, top_k=3)
        
        # Low confidence + no relevant knowledge = should say IDK
        if confidence < 0.4 and not relevant:
            return True
        
        # Very low confidence regardless
        if confidence < 0.2:
            return True
        
        # Check for tricky question patterns
        tricky_patterns = [
            r'what will happen in \d{4}',  # Future predictions
            r'exactly how many',  # Precise numbers
            r'what is my',  # Personal information
            r'what did I',  # Memory of user
            r'real-time',  # Current information
            r'live .* price',  # Stock/crypto prices
            r'current weather',  # Real-time data
        ]
        
        for pattern in tricky_patterns:
            if re.search(pattern, question.lower()):
                return True
        
        return False
    
    def _get_idk_response(self, question: str) -> str:
        """Generate an appropriate 'I don't know' response."""
        import random
        
        # Identify the topic
        topic = question.split()[:5]
        topic_str = ' '.join(topic) + "..."
        
        template = random.choice(self.idk_templates)
        return template.format(
            reason="I couldn't find reliable information on this topic.",
            topic=topic_str,
            source="a trusted source",
        )
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from a response."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Look for factual claim patterns
            has_number = bool(re.search(r'\d', sentence))
            has_entity = bool(re.search(r'[A-Z][a-z]+', sentence))
            has_definition = 'is' in sentence.lower() or 'are' in sentence.lower()
            
            if has_number or (has_entity and has_definition):
                claims.append(sentence)
        
        return claims[:10]  # Limit to 10 claims
    
    def _verify_claim(self, claim: str, knowledge_chunks: List[Dict]) -> str:
        """Verify a claim against knowledge chunks."""
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        claim_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        
        for chunk in knowledge_chunks:
            chunk_lower = chunk["text"].lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
            
            # Check word overlap
            overlap = claim_words & chunk_words
            if len(overlap) >= 3:  # Significant overlap
                # Check for contradicting terms
                negations = {'not', 'never', 'no', "n't", 'false', 'incorrect'}
                
                claim_negated = bool(negations & claim_words)
                chunk_negated = bool(negations & chunk_words)
                
                if claim_negated != chunk_negated:
                    return "contradicted"
                
                return "verified"
        
        return "uncertain"
    
    def _web_verify(self, claims: List[str]) -> Dict[str, Any]:
        """Verify claims using web search (stub for integration)."""
        # This would integrate with web search tools
        # For now, return placeholder
        return {
            "verified": [],
            "sources": [],
        }
    
    def _generate_qualified_response(
        self,
        question: str,
        original_response: str,
        confidence: float,
        uncertain: List[str],
        contradictions: List[str],
    ) -> str:
        """Generate a response with appropriate qualifications."""
        qualifiers = []
        
        if confidence < 0.5:
            qualifiers.append("I'm not entirely certain, but")
        elif uncertain:
            qualifiers.append("Based on my understanding,")
        
        if contradictions:
            qualifiers.append("(Note: Some of this may be inaccurate)")
        
        qualifier_text = ' '.join(qualifiers)
        if qualifier_text:
            return f"{qualifier_text} {original_response}"
        
        return original_response
    
    def add_knowledge(self, content: str, source: str = "user"):
        """Add new knowledge to the knowledge base."""
        self.knowledge.add_document(content, source)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fact-checker statistics."""
        return {
            "knowledge_chunks": len(self.knowledge.chunks),
            "scorer_stats": self.scorer.get_stats(),
            "config": {
                "max_context_chunks": self.config.max_context_chunks,
                "similarity_threshold": self.config.similarity_threshold,
                "use_web_search": self.config.use_web_search,
            }
        }


# Training data generator for "I don't know" responses
def generate_idk_training_data(output_path: str = "data/idk_training.txt", count: int = 50):
    """
    Generate training data to teach the AI to say "I don't know" appropriately.
    
    Creates Q&A pairs for:
    - Future predictions
    - Personal information
    - Real-time data
    - Highly technical topics outside training
    - Recent events
    """
    qa_pairs = [
        # Future predictions
        ("What will the stock market do tomorrow?", 
         "I can't predict future stock market movements. Stock prices depend on countless unpredictable factors. For investment advice, please consult a financial advisor."),
        ("Who will win the next election?",
         "I can't predict election outcomes. Election results depend on voter decisions that haven't been made yet. I can discuss polling data or historical trends if that helps."),
        ("Will it rain next month?",
         "I don't have access to long-range weather forecasts. For weather predictions, I recommend checking a weather service like weather.gov or your local forecast."),
        
        # Personal information
        ("What is my name?",
         "I don't have access to your personal information. I can only see what you share with me in our conversation."),
        ("What did I eat for breakfast?",
         "I don't know what you ate - I don't have access to information about your daily life unless you tell me."),
        ("Where do I live?",
         "I don't have access to your location or personal information. If you'd like to share that, feel free!"),
        
        # Real-time data
        ("What is the current Bitcoin price?",
         "I don't have access to real-time cryptocurrency prices. For current prices, please check an exchange like Coinbase or CoinMarketCap."),
        ("What's the weather right now?",
         "I can't access real-time weather data. Please check a weather app or website for current conditions in your area."),
        ("What's trending on Twitter today?",
         "I don't have access to real-time social media trends. Check Twitter directly for current trending topics."),
        
        # Technical limits
        ("Translate this ancient Sumerian text for me.",
         "I have very limited knowledge of ancient Sumerian. For accurate translations of ancient languages, I'd recommend consulting an academic specialist or linguistic database."),
        ("What's the exact population of my city right now?",
         "I don't have access to real-time population data. Census websites like census.gov can provide recent demographic estimates."),
        
        # Acknowledge limits
        ("Are you always right?",
         "No, I can make mistakes. I'm trained on data with a cutoff date and can sometimes generate inaccurate information. For important decisions, please verify my responses with authoritative sources."),
        ("Do you know everything?",
         "Definitely not! My knowledge has limits and a training cutoff date. I don't have real-time information, and there are many specialized topics I know little about."),
    ]
    
    # Generate more variations
    idk_templates = [
        "I'm not certain about {topic}. {suggestion}",
        "I don't have reliable information about {topic}.",
        "This is beyond my training data. {suggestion}",
        "I'd rather not guess on {topic}. {suggestion}",
    ]
    
    topics_with_suggestions = [
        ("medical diagnoses", "Please consult a healthcare professional."),
        ("legal advice", "Please consult a licensed attorney."),
        ("specific financial decisions", "Please consult a financial advisor."),
        ("your specific hardware setup", "Could you share your system specs?"),
        ("events after my training cutoff", "For recent news, check a news website."),
    ]
    
    for topic, suggestion in topics_with_suggestions:
        for template in idk_templates[:2]:
            response = template.format(topic=topic, suggestion=suggestion)
            qa_pairs.append((f"Tell me about {topic}", response))
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for q, a in qa_pairs[:count]:
            f.write(f"Q: {q}\n")
            f.write(f"A: {a}\n\n")
    
    logger.info(f"Generated {min(count, len(qa_pairs))} IDK training pairs to {output_path}")
    return output_path


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Fact Checker Demo")
    print("=" * 60)
    
    checker = FactChecker()
    
    # Add some knowledge
    checker.add_knowledge("""
    Python is a high-level programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability.
    Python supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """, source="python_facts")
    
    # Test verification
    response = "Python was created by Guido van Rossum in 1991. It's definitely the best language ever made."
    result = checker.verify_response("Who created Python?", response)
    
    print(f"\nResponse: {response}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Verified: {result.verified_facts}")
    print(f"Uncertain: {result.uncertain_claims}")
    print(f"Should qualify: {result.should_qualify}")
    
    # Test confidence scorer
    print("\n" + "=" * 60)
    scorer = ConfidenceScorer()
    
    confident = "Python was created by Guido van Rossum in 1991."
    uncertain = "I think Python might have been created around 1990, maybe by someone in the Netherlands."
    
    c1, _ = scorer.score(confident)
    c2, _ = scorer.score(uncertain)
    
    print(f"Confident response score: {c1:.2f}")
    print(f"Uncertain response score: {c2:.2f}")
    
    # Generate training data
    print("\n" + "=" * 60)
    generate_idk_training_data("data/idk_training.txt", count=20)
    print("IDK training data generated!")
