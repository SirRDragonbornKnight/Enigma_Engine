"""
Model Evaluation Suite for enigma_engine

Comprehensive evaluation tools:
- Perplexity calculation
- BLEU, ROUGE, BERTScore
- Task-specific benchmarks
- Human evaluation helpers
- Comparison utilities

Usage:
    from enigma_engine.core.evaluation import Evaluator
    
    evaluator = Evaluator(model, tokenizer)
    metrics = evaluator.evaluate_all(test_data)
"""

import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluation."""
    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    samples: Optional[list[dict[str, Any]]] = None


@dataclass
class BenchmarkResult:
    """Result from benchmark evaluation."""
    benchmark_name: str
    accuracy: float
    metrics: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    elapsed_time: float = 0.0


class Evaluator:
    """
    Main evaluation class for language models.
    
    Supports:
    - Perplexity
    - Generation quality metrics (BLEU, ROUGE)
    - Task accuracy
    - Latency measurements
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def calculate_perplexity(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> EvaluationResult:
        """
        Calculate perplexity on a dataset.
        
        Lower perplexity = better language modeling.
        """
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            if hasattr(self.tokenizer, 'encode_batch'):
                encodings = [self.tokenizer.encode(t)[:max_length] for t in batch_texts]
            else:
                encodings = [self.tokenizer.encode(t)[:max_length] for t in batch_texts]
            
            # Pad to same length
            max_len = max(len(e) for e in encodings)
            padded = []
            masks = []
            
            for e in encodings:
                pad_len = max_len - len(e)
                padded.append(e + [0] * pad_len)
                masks.append([1] * len(e) + [0] * pad_len)
            
            input_ids = torch.tensor(padded, device=self.device)
            attention_mask = torch.tensor(masks, device=self.device)
            
            # Forward pass
            outputs = self.model(input_ids)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Calculate loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            
            # Apply mask
            loss = loss.view(shift_labels.shape) * shift_mask
            
            total_loss += loss.sum().item()
            total_tokens += shift_mask.sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return EvaluationResult(
            metric_name='perplexity',
            value=perplexity,
            metadata={
                'avg_loss': avg_loss,
                'num_tokens': total_tokens,
                'num_samples': len(texts)
            }
        )
    
    def calculate_bleu(
        self,
        predictions: list[str],
        references: list[list[str]],
        max_n: int = 4
    ) -> EvaluationResult:
        """
        Calculate BLEU score.
        
        Args:
            predictions: Generated texts
            references: Reference texts (multiple refs per prediction)
            max_n: Maximum n-gram order
        """
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.lower().split()
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        def brevity_penalty(pred_len: int, ref_len: int) -> float:
            if pred_len >= ref_len:
                return 1.0
            return math.exp(1 - ref_len / pred_len)
        
        precisions = []
        pred_lengths = []
        ref_lengths = []
        
        for n in range(1, max_n + 1):
            matches = 0
            total = 0
            
            for pred, refs in zip(predictions, references):
                pred_ngrams = get_ngrams(pred, n)
                
                # Max count from any reference
                ref_counts = Counter()
                for ref in refs:
                    ref_ngrams = get_ngrams(ref, n)
                    for ng, count in ref_ngrams.items():
                        ref_counts[ng] = max(ref_counts[ng], count)
                
                # Clipped counts
                for ng, count in pred_ngrams.items():
                    matches += min(count, ref_counts[ng])
                    total += count
                
                if n == 1:
                    pred_lengths.append(len(pred.split()))
                    # Use closest reference length
                    ref_lens = [len(r.split()) for r in refs]
                    closest = min(ref_lens, key=lambda x: abs(x - len(pred.split())))
                    ref_lengths.append(closest)
            
            precision = matches / total if total > 0 else 0
            precisions.append(precision)
        
        # Calculate BLEU
        if min(precisions) > 0:
            log_precision = sum(math.log(p) for p in precisions) / max_n
            bp = brevity_penalty(sum(pred_lengths), sum(ref_lengths))
            bleu = bp * math.exp(log_precision)
        else:
            bleu = 0.0
        
        return EvaluationResult(
            metric_name='bleu',
            value=bleu * 100,  # As percentage
            metadata={
                'precisions': precisions,
                'brevity_penalty': bp if min(precisions) > 0 else 0
            }
        )
    
    def calculate_rouge(
        self,
        predictions: list[str],
        references: list[str]
    ) -> dict[str, EvaluationResult]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        """
        def lcs_length(x: list[str], y: list[str]) -> int:
            """Longest common subsequence length."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        results = {}
        
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            precision_sum = 0
            recall_sum = 0
            
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.lower().split()
                ref_tokens = ref.lower().split()
                
                if rouge_type == 'rouge1':
                    pred_set = set(pred_tokens)
                    ref_set = set(ref_tokens)
                    overlap = len(pred_set & ref_set)
                    
                elif rouge_type == 'rouge2':
                    pred_bigrams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
                    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
                    overlap = len(pred_bigrams & ref_bigrams)
                    pred_tokens = list(pred_bigrams)
                    ref_tokens = list(ref_bigrams)
                    
                else:  # rougeL
                    overlap = lcs_length(pred_tokens, ref_tokens)
                
                if len(pred_tokens) > 0:
                    precision_sum += overlap / len(pred_tokens)
                if len(ref_tokens) > 0:
                    recall_sum += overlap / len(ref_tokens)
            
            n = len(predictions)
            precision = precision_sum / n
            recall = recall_sum / n
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            results[rouge_type] = EvaluationResult(
                metric_name=rouge_type,
                value=f1 * 100,
                metadata={'precision': precision * 100, 'recall': recall * 100}
            )
        
        return results
    
    def evaluate_accuracy(
        self,
        questions: list[str],
        answers: list[str],
        generate_fn: Optional[Callable] = None,
        max_new_tokens: int = 50
    ) -> EvaluationResult:
        """
        Evaluate exact match accuracy on QA tasks.
        """
        correct = 0
        samples = []
        
        generate = generate_fn or self._default_generate
        
        for question, answer in zip(questions, answers):
            prediction = generate(question, max_new_tokens=max_new_tokens)
            
            # Normalize for comparison
            pred_normalized = prediction.lower().strip()
            ans_normalized = answer.lower().strip()
            
            is_correct = pred_normalized == ans_normalized or ans_normalized in pred_normalized
            
            if is_correct:
                correct += 1
            
            samples.append({
                'question': question,
                'expected': answer,
                'predicted': prediction,
                'correct': is_correct
            })
        
        accuracy = correct / len(questions) * 100
        
        return EvaluationResult(
            metric_name='accuracy',
            value=accuracy,
            metadata={'correct': correct, 'total': len(questions)},
            samples=samples
        )
    
    def measure_latency(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        num_warmup: int = 3
    ) -> EvaluationResult:
        """
        Measure generation latency statistics.
        """
        # Warmup
        for _ in range(num_warmup):
            _ = self._default_generate(prompts[0], max_new_tokens=10)
        
        latencies = []
        tokens_per_second = []
        
        for prompt in prompts:
            start = time.time()
            output = self._default_generate(prompt, max_new_tokens=max_new_tokens)
            elapsed = time.time() - start
            
            num_tokens = len(self.tokenizer.encode(output))
            
            latencies.append(elapsed)
            if elapsed > 0:
                tokens_per_second.append(num_tokens / elapsed)
        
        return EvaluationResult(
            metric_name='latency',
            value=sum(latencies) / len(latencies),
            metadata={
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'p50_latency': sorted(latencies)[len(latencies) // 2],
                'p99_latency': sorted(latencies)[int(len(latencies) * 0.99)],
                'avg_tokens_per_second': sum(tokens_per_second) / len(tokens_per_second)
            }
        )
    
    @torch.no_grad()
    def _default_generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Default generation function."""
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt)],
            device=self.device
        )
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.model(generated)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        output_ids = generated[0, input_ids.shape[1]:].tolist()
        return self.tokenizer.decode(output_ids)
    
    def evaluate_all(
        self,
        eval_texts: list[str],
        reference_texts: Optional[list[str]] = None,
        qa_pairs: Optional[list[tuple[str, str]]] = None
    ) -> dict[str, EvaluationResult]:
        """
        Run all applicable evaluations.
        """
        results = {}
        
        # Perplexity
        logger.info("Calculating perplexity...")
        results['perplexity'] = self.calculate_perplexity(eval_texts)
        
        # Generation metrics (if references provided)
        if reference_texts:
            logger.info("Calculating generation metrics...")
            
            # Generate outputs
            predictions = [
                self._default_generate(t[:100], max_new_tokens=50)
                for t in eval_texts
            ]
            
            results['bleu'] = self.calculate_bleu(
                predictions,
                [[r] for r in reference_texts]
            )
            
            rouge_results = self.calculate_rouge(predictions, reference_texts)
            results.update(rouge_results)
        
        # QA accuracy
        if qa_pairs:
            logger.info("Evaluating QA accuracy...")
            questions, answers = zip(*qa_pairs)
            results['accuracy'] = self.evaluate_accuracy(
                list(questions),
                list(answers)
            )
        
        # Latency
        logger.info("Measuring latency...")
        results['latency'] = self.measure_latency(eval_texts[:10])
        
        return results


class BenchmarkRunner:
    """
    Run standard benchmarks.
    
    Supports:
    - HellaSwag (commonsense)
    - MMLU (multitask)
    - TruthfulQA
    - Custom benchmarks
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = Evaluator(model, tokenizer)
    
    def run_hellaswag(
        self,
        data_path: str,
        num_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run HellaSwag benchmark (sentence completion).
        """
        # Load data
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        
        if num_samples:
            data = data[:num_samples]
        
        start_time = time.time()
        correct = 0
        
        for item in data:
            context = item['ctx']
            endings = item['endings']
            label = item['label']
            
            # Score each ending
            scores = []
            for ending in endings:
                text = context + ' ' + ending
                tokens = self.tokenizer.encode(text)
                
                with torch.no_grad():
                    input_ids = torch.tensor([tokens], device=self.evaluator.device)
                    outputs = self.model(input_ids)
                    
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    # Calculate log probability
                    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
                    token_log_probs = torch.gather(
                        log_probs,
                        2,
                        input_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                    
                    score = token_log_probs.sum().item()
                    scores.append(score)
            
            prediction = scores.index(max(scores))
            if prediction == label:
                correct += 1
        
        elapsed = time.time() - start_time
        accuracy = correct / len(data) * 100
        
        return BenchmarkResult(
            benchmark_name='hellaswag',
            accuracy=accuracy,
            num_samples=len(data),
            elapsed_time=elapsed
        )
    
    def run_mmlu(
        self,
        data_dir: str,
        subjects: Optional[list[str]] = None,
        num_shots: int = 5
    ) -> BenchmarkResult:
        """
        Run MMLU benchmark (multi-task).
        """
        # Find all subject files
        all_subjects = []
        for f in os.listdir(data_dir):
            if f.endswith('_test.csv'):
                subject = f.replace('_test.csv', '')
                if subjects is None or subject in subjects:
                    all_subjects.append(subject)
        
        start_time = time.time()
        total_correct = 0
        total_samples = 0
        per_category = {}
        
        for subject in all_subjects:
            test_path = os.path.join(data_dir, f'{subject}_test.csv')
            
            with open(test_path) as f:
                lines = f.readlines()
            
            correct = 0
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                question = parts[0]
                choices = parts[1:5]
                answer = parts[5]
                
                # Score each choice
                scores = []
                for choice in choices:
                    text = f"Question: {question}\nAnswer: {choice}"
                    score = self._score_text(text)
                    scores.append(score)
                
                prediction = ['A', 'B', 'C', 'D'][scores.index(max(scores))]
                if prediction == answer:
                    correct += 1
            
            per_category[subject] = correct / len(lines) * 100 if lines else 0
            total_correct += correct
            total_samples += len(lines)
        
        elapsed = time.time() - start_time
        accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        return BenchmarkResult(
            benchmark_name='mmlu',
            accuracy=accuracy,
            per_category=per_category,
            num_samples=total_samples,
            elapsed_time=elapsed
        )
    
    @torch.no_grad()
    def _score_text(self, text: str) -> float:
        """Score text by log probability."""
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], device=self.evaluator.device)
        
        outputs = self.model(input_ids)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            2,
            input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs.sum().item()


def compare_models(
    models: list[tuple[str, torch.nn.Module]],
    tokenizer: Any,
    eval_texts: list[str]
) -> dict[str, dict[str, float]]:
    """
    Compare multiple models on the same evaluation data.
    
    Args:
        models: List of (name, model) tuples
        tokenizer: Shared tokenizer
        eval_texts: Evaluation texts
    
    Returns:
        Comparison results by model and metric
    """
    results = {}
    
    for name, model in models:
        logger.info(f"Evaluating model: {name}")
        
        evaluator = Evaluator(model, tokenizer)
        model_results = evaluator.evaluate_all(eval_texts)
        
        results[name] = {
            metric: result.value
            for metric, result in model_results.items()
        }
    
    return results
