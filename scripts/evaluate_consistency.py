#!/usr/bin/env python3
"""
Script to evaluate explanation consistency using ExPara-Sim metrics.

Usage:
    python scripts/evaluate_consistency.py --explanations data/explanations/ --output results/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.models import (
    Question, Paraphrase, Explanation, ExplanationPair,
    load_jsonl, save_jsonl
)
from evaluation.pipeline import EvaluationPipeline, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(
    paraphrases_dir: str,
    explanations_dir: str,
    dataset: str = None
) -> Dict:
    """Load all necessary data for evaluation."""
    paraphrases_path = Path(paraphrases_dir)
    explanations_path = Path(explanations_dir)
    
    data = {
        "questions": {},
        "paraphrases": {},
        "explanations_original": {},
        "explanations_paraphrase": {}
    }
    
    # Find question files
    question_files = list(paraphrases_path.glob("*_questions.jsonl"))
    if dataset:
        question_files = [f for f in question_files if dataset in f.name]
    
    for qf in question_files:
        dataset_name = qf.stem.replace("_questions", "")
        logger.info(f"Loading {dataset_name}...")
        
        # Load questions
        questions = load_jsonl(qf, Question)
        for q in questions:
            data["questions"][q.id] = q
        
        # Load paraphrases
        pf = qf.parent / f"{dataset_name}_paraphrases.jsonl"
        if pf.exists():
            paraphrases = load_jsonl(pf, Paraphrase)
            for p in paraphrases:
                if p.original_question_id not in data["paraphrases"]:
                    data["paraphrases"][p.original_question_id] = []
                data["paraphrases"][p.original_question_id].append(p)
        
        # Load original explanations
        orig_files = list(explanations_path.glob(f"{dataset_name}_*_original_explanations.jsonl"))
        for ef in orig_files:
            explanations = load_jsonl(ef, Explanation)
            for e in explanations:
                data["explanations_original"][e.question_id] = e
        
        # Load paraphrase explanations
        para_files = list(explanations_path.glob(f"{dataset_name}_*_paraphrase_explanations.jsonl"))
        for ef in para_files:
            explanations = load_jsonl(ef, Explanation)
            for e in explanations:
                data["explanations_paraphrase"][e.question_id] = e
    
    return data


def print_results(result, analysis):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*60)
    print("EXPARA EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nModel: {result.model}")
    print(f"Prompt Type: {result.prompt_type}")
    print(f"Temperature: {result.temperature}")
    print(f"Number of Pairs: {result.num_pairs}")
    
    print("\n--- Overall Metrics ---")
    print(f"ExPara-Sim Score: {result.avg_expara_sim:.4f} (±{result.std_expara_sim:.4f})")
    print(f"Answer Consistency: {result.answer_consistency_rate:.2%}")
    
    print("\n--- Per-Metric Breakdown ---")
    print(f"  Semantic Similarity: {result.avg_semantic_sim:.4f}")
    print(f"  Entailment Symmetric: {result.avg_entailment_sym:.4f}")
    print(f"  Step Alignment:      {result.avg_step_align:.4f}")
    print(f"  Fact Overlap:        {result.avg_fact_overlap:.4f}")
    print(f"  LLM Judge:           {result.avg_llm_judge:.4f}")
    
    print("\n--- By Task Type ---")
    for task, stats in result.by_task_type.items():
        print(f"  {task}:")
        print(f"    ExPara-Sim: {stats['avg_expara_sim']:.4f}")
        print(f"    Answer Consistency: {stats['answer_consistency']:.2%}")
        print(f"    N = {stats['num_pairs']}")
    
    print("\n--- By Paraphrase Type ---")
    for para_type, stats in result.by_paraphrase_type.items():
        print(f"  {para_type}: {stats['avg_expara_sim']:.4f} (N={stats['num_pairs']})")
    
    print("\n--- Divergence Analysis ---")
    print(f"Same-answer pairs: {analysis['total_same_answer_pairs']}")
    print(f"Divergent explanations: {analysis['divergent_pairs']} ({analysis['divergence_rate']:.2%})")
    
    if analysis.get("divergence_by_task"):
        print("\nDivergence by task type:")
        for task, stats in analysis["divergence_by_task"].items():
            print(f"  {task}: {stats['rate']:.2%} ({stats['divergent']}/{stats['total']})")
    
    if analysis.get("example_divergent"):
        print("\n--- Example Divergent Pairs ---")
        for i, ex in enumerate(analysis["example_divergent"][:3], 1):
            print(f"\n[{i}] ExPara-Sim: {ex['expara_sim']:.3f}")
            print(f"Q: {ex['question'][:100]}...")
            print(f"P: {ex['paraphrase'][:100]}...")
            print(f"Answer: {ex['answer']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate explanation consistency")
    
    parser.add_argument(
        "--paraphrases",
        type=str,
        default="data/paraphrases/",
        help="Directory with questions and paraphrases"
    )
    parser.add_argument(
        "--explanations",
        type=str,
        required=True,
        help="Directory with explanations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unknown",
        help="Model name for results"
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Include LLM judge metric (requires API)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding models"
    )
    parser.add_argument(
        "--divergence_threshold",
        type=float,
        default=0.5,
        help="Threshold for divergence analysis"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data = load_data(args.paraphrases, args.explanations, args.dataset)
    
    logger.info(f"Loaded:")
    logger.info(f"  Questions: {len(data['questions'])}")
    logger.info(f"  Paraphrases: {sum(len(v) for v in data['paraphrases'].values())}")
    logger.info(f"  Original explanations: {len(data['explanations_original'])}")
    logger.info(f"  Paraphrase explanations: {len(data['explanations_paraphrase'])}")
    
    # Create evaluation config
    config = EvaluationConfig(
        models=[args.model],
        use_llm_judge=args.use_llm_judge,
        device=args.device,
        output_dir=args.output
    )
    
    # Initialize pipeline
    logger.info("Initializing evaluation pipeline...")
    pipeline = EvaluationPipeline(config)
    
    # Create pairs
    logger.info("Creating explanation pairs...")
    pairs = pipeline.create_explanation_pairs(
        data["questions"],
        data["paraphrases"],
        data["explanations_original"],
        data["explanations_paraphrase"]
    )
    logger.info(f"Created {len(pairs)} pairs")
    
    if not pairs:
        logger.error("No pairs created. Check that explanations match questions/paraphrases.")
        return
    
    # Evaluate
    logger.info("Evaluating pairs...")
    pairs = pipeline.evaluate_pairs(pairs)
    
    # Aggregate results
    logger.info("Computing aggregate results...")
    result = pipeline.compute_aggregate_results(
        pairs,
        model=args.model,
        prompt_type="zero_shot_cot",
        temperature=0.0
    )
    
    # Analyze divergence
    logger.info("Analyzing divergence patterns...")
    analysis = pipeline.analyze_divergence_patterns(pairs, args.divergence_threshold)
    
    # Save results
    pipeline.save_results(pairs, result)
    
    # Save analysis
    analysis_path = os.path.join(args.output, f"divergence_analysis_{args.model}.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Saved analysis to {analysis_path}")
    
    # Print results
    print_results(result, analysis)


if __name__ == "__main__":
    main()
