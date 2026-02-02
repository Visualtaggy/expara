"""
ExPara-Sim Metrics
Core metrics for measuring explanation consistency.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a single metric computation."""
    name: str
    score: float
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseMetric(ABC):
    """Base class for all ExPara metrics."""
    
    name: str = "base"
    
    @abstractmethod
    def compute(self, explanation1: str, explanation2: str) -> MetricResult:
        """Compute the metric between two explanations."""
        pass
    
    def compute_batch(
        self,
        explanations1: List[str],
        explanations2: List[str]
    ) -> List[MetricResult]:
        """Compute metric for a batch of explanation pairs."""
        return [
            self.compute(e1, e2) 
            for e1, e2 in zip(explanations1, explanations2)
        ]


class SemanticSimilarityMetric(BaseMetric):
    """
    Computes semantic similarity using sentence embeddings.
    Uses cosine similarity between embedding vectors.
    """
    
    name = "semantic_sim"
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        self._load_model()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def compute(self, explanation1: str, explanation2: str) -> MetricResult:
        """Compute semantic similarity between two explanations."""
        emb1 = self.get_embedding(explanation1)
        emb2 = self.get_embedding(explanation2)
        score = self.cosine_similarity(emb1, emb2)
        
        return MetricResult(
            name=self.name,
            score=float(score),
            details={"embedding_dim": len(emb1)}
        )
    
    def compute_batch(
        self,
        explanations1: List[str],
        explanations2: List[str]
    ) -> List[MetricResult]:
        """Efficient batch computation."""
        self._load_model()
        
        # Get all embeddings at once
        all_texts = explanations1 + explanations2
        all_embeddings = self.get_embeddings_batch(all_texts)
        
        n = len(explanations1)
        emb1 = all_embeddings[:n]
        emb2 = all_embeddings[n:]
        
        results = []
        for e1, e2 in zip(emb1, emb2):
            score = self.cosine_similarity(e1, e2)
            results.append(MetricResult(
                name=self.name,
                score=float(score)
            ))
        
        return results


class EntailmentSymmetricMetric(BaseMetric):
    """
    Bidirectional entailment metric.
    Computes average of P(explanation1 entails explanation2) and P(explanation2 entails explanation1).
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the NLI model."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self._model.eval()
    
    def get_entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Get probability that premise entails hypothesis."""
        import torch
        
        self._load_model()
        
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Get entailment probability
            # DeBERTa MNLI labels: 0=contradiction, 1=neutral, 2=entailment
            entailment_prob = probs[0, 2].item()
        
        return entailment_prob
    
    def compute(self, explanation1: str, explanation2: str) -> MetricResult:
        """Compute bidirectional entailment score."""
        prob_1_to_2 = self.get_entailment_prob(explanation1, explanation2)
        prob_2_to_1 = self.get_entailment_prob(explanation2, explanation1)
        
        score = (prob_1_to_2 + prob_2_to_1) / 2
        
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "prob_1_entails_2": prob_1_to_2,
                "prob_2_entails_1": prob_2_to_1
            }
        )


class StepAlignmentMetric(BaseMetric):
    """
    Measures alignment of reasoning steps using optimal transport.
    Computes the cost of aligning steps from one explanation to another.
    """
    
    name = "step_align"
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cuda",
        step_delimiter: str = "\n"
    ):
        self.embedding_model = embedding_model
        self.device = device
        self.step_delimiter = step_delimiter
        self._embedder = None
    
    def _load_embedder(self):
        """Lazy load sentence embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model, device=self.device)
    
    def extract_steps(self, explanation: str) -> List[str]:
        """Extract reasoning steps from explanation."""
        import re
        
        # Try numbered steps first
        numbered_pattern = r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\n\n|$)"
        matches = re.findall(numbered_pattern, explanation, re.MULTILINE | re.DOTALL)
        
        if matches:
            return [m.strip() for m in matches if m.strip()]
        
        # Fall back to line-based splitting
        lines = explanation.split(self.step_delimiter)
        steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        
        return steps if steps else [explanation]
    
    def compute_transport_distance(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> float:
        """Compute optimal transport distance between two sets of embeddings."""
        import ot  # POT library
        
        n1, n2 = len(embeddings1), len(embeddings2)
        
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Compute cost matrix (cosine distance)
        # Normalize embeddings
        emb1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        emb2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm.T)
        cost_matrix = 1 - similarity
        
        # Uniform distributions
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        
        # Compute EMD
        transport_cost = ot.emd2(a, b, cost_matrix)
        
        # Convert distance to similarity (0 to 1)
        similarity_score = 1 - transport_cost
        
        return max(0.0, min(1.0, similarity_score))
    
    def compute(self, explanation1: str, explanation2: str) -> MetricResult:
        """Compute step alignment score."""
        self._load_embedder()
        
        # Extract steps
        steps1 = self.extract_steps(explanation1)
        steps2 = self.extract_steps(explanation2)
        
        # Get embeddings
        if not steps1 or not steps2:
            return MetricResult(name=self.name, score=0.0)
        
        emb1 = self._embedder.encode(steps1, convert_to_numpy=True)
        emb2 = self._embedder.encode(steps2, convert_to_numpy=True)
        
        # Compute optimal transport
        score = self.compute_transport_distance(emb1, emb2)
        
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "num_steps_1": len(steps1),
                "num_steps_2": len(steps2)
            }
        )


class FactOverlapMetric(BaseMetric):
    """
    Measures overlap of factual claims between explanations.
    Uses Jaccard similarity of extracted claims.
    """
    
    name = "fact_overlap"
    
    def __init__(
        self,
        extraction_method: str = "regex",  # or "llm"
        llm_model: Optional[str] = None
    ):
        self.extraction_method = extraction_method
        self.llm_model = llm_model
    
    def extract_facts_regex(self, text: str) -> List[str]:
        """Extract factual claims using regex patterns."""
        import re
        
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            
            # Look for factual patterns
            # Numbers and quantities
            if re.search(r'\d+', sent):
                facts.append(sent)
            # Copula statements (X is Y)
            elif re.search(r'\b(is|are|was|were|equals?|means?)\b', sent, re.I):
                facts.append(sent)
            # Causal statements
            elif re.search(r'\b(because|therefore|thus|so|hence|since)\b', sent, re.I):
                facts.append(sent)
        
        return facts
    
    def normalize_fact(self, fact: str) -> str:
        """Normalize a fact for comparison."""
        import re
        
        # Lowercase
        fact = fact.lower()
        # Remove extra whitespace
        fact = re.sub(r'\s+', ' ', fact).strip()
        # Remove punctuation except numbers
        fact = re.sub(r'[^\w\s\d]', '', fact)
        
        return fact
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union
    
    def compute(self, explanation1: str, explanation2: str) -> MetricResult:
        """Compute fact overlap score."""
        # Extract facts
        facts1 = self.extract_facts_regex(explanation1)
        facts2 = self.extract_facts_regex(explanation2)
        
        # Normalize
        normalized1 = set(self.normalize_fact(f) for f in facts1)
        normalized2 = set(self.normalize_fact(f) for f in facts2)
        
        # Remove empty strings
        normalized1.discard('')
        normalized2.discard('')
        
        # Compute Jaccard similarity
        score = self.jaccard_similarity(normalized1, normalized2)
        
        return MetricResult(
            name=self.name,
            score=score,
            details={
                "num_facts_1": len(facts1),
                "num_facts_2": len(facts2),
                "overlap_count": len(normalized1 & normalized2)
            }
        )


class LLMJudgeMetric(BaseMetric):
    """
    Uses an LLM to judge explanation consistency.
    Asks whether two explanations convey the same reasoning.
    """
    
    name = "llm_judge"
    
    JUDGE_PROMPT = """You are evaluating whether two explanations convey the same reasoning for answering a question.

Question: {question}

Explanation 1:
{explanation1}

Explanation 2:
{explanation2}

Consider:
1. Do both explanations use the same key reasoning steps?
2. Do they rely on the same facts or premises?
3. Do they reach the conclusion through similar logical paths?

Rate the consistency of reasoning on a scale from 0 to 1:
- 0.0: Completely different reasoning
- 0.25: Mostly different, some overlap
- 0.5: Partially similar reasoning
- 0.75: Mostly similar, minor differences
- 1.0: Essentially the same reasoning

Respond with ONLY a number between 0 and 1."""

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        num_samples: int = 1
    ):
        self.model = model
        self.provider = provider
        self.num_samples = num_samples
        self._client = None
    
    def _load_client(self):
        """Lazy load the LLM client."""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
    
    def get_judgment(
        self,
        question: str,
        explanation1: str,
        explanation2: str
    ) -> float:
        """Get a single judgment from the LLM."""
        import re
        
        self._load_client()
        
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            explanation1=explanation1,
            explanation2=explanation2
        )
        
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                result = response.choices[0].message.content.strip()
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text.strip()
            
            # Parse the score
            match = re.search(r'(\d+\.?\d*)', result)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            
            return 0.5  # Default if parsing fails
            
        except Exception as e:
            logger.error(f"LLM judge error: {e}")
            return 0.5
    
    def compute(
        self,
        explanation1: str,
        explanation2: str,
        question: str = ""
    ) -> MetricResult:
        """Compute LLM judge score."""
        scores = []
        for _ in range(self.num_samples):
            score = self.get_judgment(question, explanation1, explanation2)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        return MetricResult(
            name=self.name,
            score=float(avg_score),
            details={
                "individual_scores": scores,
                "std": float(np.std(scores)) if len(scores) > 1 else 0.0
            }
        )


class ExParaSimScore:
    """
    Composite ExPara-Sim score combining all metrics.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        use_llm_judge: bool = True
    ):
        # Default weights (can be learned from human correlation)
        self.weights = weights or {
            "semantic_sim": 0.25,
            "entailment_sym": 0.25,
            "step_align": 0.20,
            "fact_overlap": 0.15,
            "llm_judge": 0.15
        }
        
        # Initialize metrics
        self.metrics = {
            "semantic_sim": SemanticSimilarityMetric(device=device),
            "entailment_sym": EntailmentSymmetricMetric(device=device),
            "step_align": StepAlignmentMetric(device=device),
            "fact_overlap": FactOverlapMetric()
        }
        
        if use_llm_judge:
            self.metrics["llm_judge"] = LLMJudgeMetric()
        else:
            # Remove from weights if not using
            self.weights.pop("llm_judge", None)
            # Renormalize weights
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def compute(
        self,
        explanation1: str,
        explanation2: str,
        question: str = ""
    ) -> Tuple[float, Dict[str, MetricResult]]:
        """Compute the composite ExPara-Sim score."""
        results = {}
        
        for name, metric in self.metrics.items():
            if name == "llm_judge":
                result = metric.compute(explanation1, explanation2, question)
            else:
                result = metric.compute(explanation1, explanation2)
            results[name] = result
        
        # Compute weighted score
        composite_score = sum(
            self.weights.get(name, 0) * result.score
            for name, result in results.items()
        )
        
        return composite_score, results
    
    def compute_batch(
        self,
        explanations1: List[str],
        explanations2: List[str],
        questions: Optional[List[str]] = None
    ) -> List[Tuple[float, Dict[str, MetricResult]]]:
        """Compute scores for a batch of explanation pairs."""
        if questions is None:
            questions = [""] * len(explanations1)
        
        results = []
        for e1, e2, q in zip(explanations1, explanations2, questions):
            score, metrics = self.compute(e1, e2, q)
            results.append((score, metrics))
        
        return results


if __name__ == "__main__":
    # Test metrics
    exp1 = """Let's solve this step by step:
    1. We need to find the distance traveled
    2. Distance = Speed × Time
    3. Speed is 60 mph
    4. Time is 2 hours
    5. Distance = 60 × 2 = 120 miles
    Therefore, the train travels 120 miles."""
    
    exp2 = """To calculate the distance:
    First, recall that distance equals rate multiplied by time.
    The rate is 60 miles per hour.
    The time duration is 2 hours.
    So, 60 mph × 2 h = 120 miles.
    The answer is 120 miles."""
    
    exp3 = """The capital of France is Paris. It's a beautiful city known for the Eiffel Tower."""
    
    print("Testing ExPara-Sim Metrics")
    print("="*50)
    
    # Test individual metrics (without loading heavy models)
    fact_metric = FactOverlapMetric()
    
    print("\nFact Overlap (similar explanations):")
    result = fact_metric.compute(exp1, exp2)
    print(f"  Score: {result.score:.3f}")
    print(f"  Details: {result.details}")
    
    print("\nFact Overlap (different explanations):")
    result = fact_metric.compute(exp1, exp3)
    print(f"  Score: {result.score:.3f}")
    print(f"  Details: {result.details}")
