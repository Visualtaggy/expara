"""
ExPara Evaluation Pipeline
Main pipeline for evaluating explanation consistency across models.
"""

import asyncio
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np

from data.models import (
    Question, Paraphrase, Explanation, ExplanationPair,
    EvaluationResult, TaskType, ParaphraseType, save_jsonl, load_jsonl
)
from metrics.expara_sim import ExParaSimScore

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation pipeline."""
    # Models to evaluate
    models: List[str] = field(default_factory=lambda: ["gpt-4o"])
    
    # Prompting settings
    prompt_types: List[str] = field(default_factory=lambda: ["zero_shot_cot"])
    temperatures: List[float] = field(default_factory=lambda: [0.0])
    
    # Metric settings
    use_llm_judge: bool = True
    device: str = "cuda"
    
    # Output settings
    output_dir: str = "results/"
    save_explanations: bool = True
    save_pairs: bool = True
    
    # Processing
    batch_size: int = 32
    max_concurrent: int = 10


class EvaluationPipeline:
    """
    Main pipeline for evaluating explanation consistency.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.scorer = ExParaSimScore(
            device=config.device,
            use_llm_judge=config.use_llm_judge
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def create_explanation_pairs(
        self,
        questions: Dict[str, Question],
        paraphrases: Dict[str, List[Paraphrase]],
        explanations_original: Dict[str, Explanation],
        explanations_paraphrase: Dict[str, Explanation]
    ) -> List[ExplanationPair]:
        """Create explanation pairs for evaluation."""
        pairs = []
        
        for q_id, question in questions.items():
            if q_id not in paraphrases:
                continue
            
            original_exp = explanations_original.get(q_id)
            if original_exp is None:
                continue
            
            for paraphrase in paraphrases[q_id]:
                para_exp = explanations_paraphrase.get(paraphrase.id)
                if para_exp is None:
                    continue
                
                pair = ExplanationPair(
                    id=f"{q_id}_{paraphrase.id}",
                    original_question=question,
                    paraphrase=paraphrase,
                    explanation_original=original_exp,
                    explanation_paraphrase=para_exp,
                    same_answer=self._check_same_answer(
                        original_exp.final_answer,
                        para_exp.final_answer
                    )
                )
                pairs.append(pair)
        
        return pairs
    
    def _check_same_answer(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are equivalent."""
        import re
        
        # Normalize
        a1 = answer1.lower().strip()
        a2 = answer2.lower().strip()
        
        # Direct match
        if a1 == a2:
            return True
        
        # One contains the other
        if a1 in a2 or a2 in a1:
            return True
        
        # Numeric comparison
        nums1 = re.findall(r'[\d,]+\.?\d*', a1)
        nums2 = re.findall(r'[\d,]+\.?\d*', a2)
        
        if nums1 and nums2:
            try:
                val1 = float(nums1[-1].replace(',', ''))
                val2 = float(nums2[-1].replace(',', ''))
                if abs(val1 - val2) < 0.01:
                    return True
            except ValueError:
                pass
        
        # Yes/No comparison
        yes_words = {'yes', 'true', 'correct', 'right'}
        no_words = {'no', 'false', 'incorrect', 'wrong'}
        
        a1_yes = any(w in a1 for w in yes_words)
        a1_no = any(w in a1 for w in no_words)
        a2_yes = any(w in a2 for w in yes_words)
        a2_no = any(w in a2 for w in no_words)
        
        if (a1_yes and a2_yes) or (a1_no and a2_no):
            return True
        
        # Multiple choice
        mc1 = re.match(r'^([a-d])\b', a1)
        mc2 = re.match(r'^([a-d])\b', a2)
        if mc1 and mc2:
            return mc1.group(1) == mc2.group(1)
        
        return False
    
    def evaluate_pairs(
        self,
        pairs: List[ExplanationPair],
        show_progress: bool = True
    ) -> List[ExplanationPair]:
        """Evaluate all explanation pairs."""
        iterator = tqdm(pairs, desc="Evaluating pairs") if show_progress else pairs
        
        for pair in iterator:
            # Compute ExPara-Sim score
            score, metrics = self.scorer.compute(
                pair.explanation_original.reasoning,
                pair.explanation_paraphrase.reasoning,
                pair.original_question.text
            )
            
            pair.expara_sim_score = score
            pair.metrics = {name: result.score for name, result in metrics.items()}
        
        return pairs
    
    def compute_aggregate_results(
        self,
        pairs: List[ExplanationPair],
        model: str,
        prompt_type: str,
        temperature: float
    ) -> EvaluationResult:
        """Compute aggregate statistics from evaluated pairs."""
        if not pairs:
            raise ValueError("No pairs to aggregate")
        
        # Collect scores
        expara_scores = [p.expara_sim_score for p in pairs if p.expara_sim_score is not None]
        
        # Per-metric scores
        metric_names = ["semantic_sim", "entailment_sym", "step_align", "fact_overlap", "llm_judge"]
        metric_scores = {name: [] for name in metric_names}
        
        for pair in pairs:
            for name in metric_names:
                if name in pair.metrics:
                    metric_scores[name].append(pair.metrics[name])
        
        # Answer consistency
        answer_consistent = sum(1 for p in pairs if p.same_answer)
        answer_consistency_rate = answer_consistent / len(pairs)
        
        # By task type
        by_task_type = {}
        for task_type in TaskType:
            task_pairs = [p for p in pairs if p.original_question.task_type == task_type]
            if task_pairs:
                task_scores = [p.expara_sim_score for p in task_pairs if p.expara_sim_score]
                by_task_type[task_type.value] = {
                    "avg_expara_sim": float(np.mean(task_scores)) if task_scores else 0.0,
                    "num_pairs": len(task_pairs),
                    "answer_consistency": sum(1 for p in task_pairs if p.same_answer) / len(task_pairs)
                }
        
        # By paraphrase type
        by_paraphrase_type = {}
        for para_type in ParaphraseType:
            para_pairs = [p for p in pairs if p.paraphrase.paraphrase_type == para_type]
            if para_pairs:
                para_scores = [p.expara_sim_score for p in para_pairs if p.expara_sim_score]
                by_paraphrase_type[para_type.value] = {
                    "avg_expara_sim": float(np.mean(para_scores)) if para_scores else 0.0,
                    "num_pairs": len(para_pairs)
                }
        
        return EvaluationResult(
            model=model,
            prompt_type=prompt_type,
            temperature=temperature,
            num_pairs=len(pairs),
            avg_expara_sim=float(np.mean(expara_scores)) if expara_scores else 0.0,
            std_expara_sim=float(np.std(expara_scores)) if expara_scores else 0.0,
            avg_semantic_sim=float(np.mean(metric_scores["semantic_sim"])) if metric_scores["semantic_sim"] else 0.0,
            avg_entailment_sym=float(np.mean(metric_scores["entailment_sym"])) if metric_scores["entailment_sym"] else 0.0,
            avg_step_align=float(np.mean(metric_scores["step_align"])) if metric_scores["step_align"] else 0.0,
            avg_fact_overlap=float(np.mean(metric_scores["fact_overlap"])) if metric_scores["fact_overlap"] else 0.0,
            avg_llm_judge=float(np.mean(metric_scores["llm_judge"])) if metric_scores["llm_judge"] else 0.0,
            answer_consistency_rate=answer_consistency_rate,
            by_task_type=by_task_type,
            by_paraphrase_type=by_paraphrase_type
        )
    
    def save_results(
        self,
        pairs: List[ExplanationPair],
        result: EvaluationResult,
        suffix: str = ""
    ):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.model.replace('/', '_')}_{result.prompt_type}_T{result.temperature}_{timestamp}{suffix}"
        
        # Save pairs
        if self.config.save_pairs:
            pairs_path = os.path.join(self.config.output_dir, f"pairs_{base_name}.jsonl")
            save_jsonl(pairs, pairs_path)
            logger.info(f"Saved pairs to {pairs_path}")
        
        # Save aggregate results
        results_path = os.path.join(self.config.output_dir, f"results_{base_name}.json")
        with open(results_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved results to {results_path}")
    
    def analyze_divergence_patterns(
        self,
        pairs: List[ExplanationPair],
        threshold: float = 0.5
    ) -> Dict:
        """Analyze patterns in explanation divergence."""
        # Filter to same-answer pairs (most interesting for divergence analysis)
        same_answer_pairs = [p for p in pairs if p.same_answer]
        
        if not same_answer_pairs:
            return {"error": "No same-answer pairs to analyze"}
        
        # Find high-divergence pairs (low consistency despite same answer)
        divergent = [p for p in same_answer_pairs if p.expara_sim_score and p.expara_sim_score < threshold]
        consistent = [p for p in same_answer_pairs if p.expara_sim_score and p.expara_sim_score >= threshold]
        
        analysis = {
            "total_same_answer_pairs": len(same_answer_pairs),
            "divergent_pairs": len(divergent),
            "consistent_pairs": len(consistent),
            "divergence_rate": len(divergent) / len(same_answer_pairs) if same_answer_pairs else 0,
            
            # By task type
            "divergence_by_task": {},
            
            # By paraphrase type
            "divergence_by_paraphrase": {},
            
            # Example divergent pairs
            "example_divergent": []
        }
        
        # Divergence by task type
        for task_type in TaskType:
            task_divergent = [p for p in divergent if p.original_question.task_type == task_type]
            task_total = [p for p in same_answer_pairs if p.original_question.task_type == task_type]
            if task_total:
                analysis["divergence_by_task"][task_type.value] = {
                    "divergent": len(task_divergent),
                    "total": len(task_total),
                    "rate": len(task_divergent) / len(task_total)
                }
        
        # Divergence by paraphrase type
        for para_type in ParaphraseType:
            para_divergent = [p for p in divergent if p.paraphrase.paraphrase_type == para_type]
            para_total = [p for p in same_answer_pairs if p.paraphrase.paraphrase_type == para_type]
            if para_total:
                analysis["divergence_by_paraphrase"][para_type.value] = {
                    "divergent": len(para_divergent),
                    "total": len(para_total),
                    "rate": len(para_divergent) / len(para_total)
                }
        
        # Add example divergent pairs
        sorted_divergent = sorted(divergent, key=lambda p: p.expara_sim_score or 0)
        for pair in sorted_divergent[:5]:
            analysis["example_divergent"].append({
                "question": pair.original_question.text[:200],
                "paraphrase": pair.paraphrase.text[:200],
                "answer": pair.explanation_original.final_answer,
                "expara_sim": pair.expara_sim_score,
                "explanation_1_snippet": pair.explanation_original.reasoning[:300],
                "explanation_2_snippet": pair.explanation_paraphrase.reasoning[:300]
            })
        
        return analysis


def run_full_evaluation(
    questions: Dict[str, Question],
    paraphrases: Dict[str, List[Paraphrase]],
    explanations_original: Dict[str, Explanation],
    explanations_paraphrase: Dict[str, Explanation],
    config: Optional[EvaluationConfig] = None
) -> Tuple[List[ExplanationPair], EvaluationResult, Dict]:
    """Run the full evaluation pipeline."""
    if config is None:
        config = EvaluationConfig()
    
    pipeline = EvaluationPipeline(config)
    
    # Create pairs
    logger.info("Creating explanation pairs...")
    pairs = pipeline.create_explanation_pairs(
        questions, paraphrases,
        explanations_original, explanations_paraphrase
    )
    logger.info(f"Created {len(pairs)} pairs")
    
    # Evaluate
    logger.info("Evaluating pairs...")
    pairs = pipeline.evaluate_pairs(pairs)
    
    # Aggregate
    logger.info("Computing aggregate results...")
    result = pipeline.compute_aggregate_results(
        pairs,
        model=config.models[0] if config.models else "unknown",
        prompt_type=config.prompt_types[0] if config.prompt_types else "unknown",
        temperature=config.temperatures[0] if config.temperatures else 0.0
    )
    
    # Analyze divergence
    logger.info("Analyzing divergence patterns...")
    divergence_analysis = pipeline.analyze_divergence_patterns(pairs)
    
    # Save
    pipeline.save_results(pairs, result)
    
    return pairs, result, divergence_analysis


if __name__ == "__main__":
    # Demo with mock data
    logging.basicConfig(level=logging.INFO)
    
    # Create mock data for testing
    from data.models import Question, Paraphrase, Explanation, TaskType, ParaphraseType
    
    questions = {
        "q1": Question(
            id="q1",
            text="What is 2 + 2?",
            answer="4",
            task_type=TaskType.MATH,
            source_dataset="test"
        )
    }
    
    paraphrases = {
        "q1": [
            Paraphrase(
                id="p1",
                original_question_id="q1",
                text="Calculate the sum of 2 and 2",
                paraphrase_type=ParaphraseType.LEXICAL,
                generator_model="test"
            )
        ]
    }
    
    explanations_original = {
        "q1": Explanation(
            id="e1",
            question_id="q1",
            question_text="What is 2 + 2?",
            reasoning="Let me add these numbers. 2 + 2 = 4. The answer is 4.",
            final_answer="4",
            model="test",
            prompt_type="zero_shot_cot",
            temperature=0.0
        )
    }
    
    explanations_paraphrase = {
        "p1": Explanation(
            id="e2",
            question_id="p1",
            question_text="Calculate the sum of 2 and 2",
            reasoning="To find the sum: 2 plus 2 equals 4. Therefore, the answer is 4.",
            final_answer="4",
            model="test",
            prompt_type="zero_shot_cot",
            temperature=0.0
        )
    }
    
    print("Running evaluation pipeline demo...")
    config = EvaluationConfig(
        use_llm_judge=False,  # Skip LLM judge for demo
        device="cpu"
    )
    
    # Note: Full run requires model loading
    # pairs, result, analysis = run_full_evaluation(
    #     questions, paraphrases,
    #     explanations_original, explanations_paraphrase,
    #     config
    # )
    
    print("Demo complete!")
