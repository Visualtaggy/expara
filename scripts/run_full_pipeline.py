#!/usr/bin/env python3
"""
Run the complete ExPara evaluation pipeline.

This script runs all stages:
1. Load datasets
2. Generate paraphrases
3. Generate explanations
4. Evaluate consistency
5. Analyze results

Usage:
    python scripts/run_full_pipeline.py --models Qwen/Qwen2.5-7B-Instruct --num_samples 250
    python scripts/run_full_pipeline.py --models Qwen/Qwen2.5-14B-Instruct --num_samples 250 --resume
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DatasetLoader
from data.paraphrase_generator import ParaphraseGenerator, ParaphraseValidator
from data.models import save_jsonl, load_jsonl, Question, Paraphrase, Explanation
from evaluation.explanation_generator import ExplanationGenerator, GenerationConfig
from evaluation.pipeline import EvaluationPipeline, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExParaPipeline:
    """Complete ExPara benchmark pipeline."""
    
    def __init__(
        self,
        output_dir: str = "results",
        data_dir: str = "data",
        device: str = "cuda"
    ):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.device = device
        
        # Create directories
        self.paraphrases_dir = self.data_dir / "paraphrases"
        self.explanations_dir = self.data_dir / "explanations"
        self.results_dir = self.output_dir
        
        for d in [self.paraphrases_dir, self.explanations_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def stage1_load_datasets(
        self,
        datasets: list = None,
        num_samples: int = 1000
    ) -> dict:
        """Stage 1: Load source datasets."""
        logger.info("="*50)
        logger.info("STAGE 1: Loading datasets")
        logger.info("="*50)
        
        loader = DatasetLoader()
        
        if datasets is None or "all" in datasets:
            questions_dict = loader.load_all(num_samples)
        else:
            questions_dict = {}
            for ds in datasets:
                load_func = getattr(loader, f"load_{ds}", None)
                if load_func:
                    questions_dict[ds] = load_func(num_samples)
        
        # Save questions
        all_questions = {}
        for ds_name, questions in questions_dict.items():
            path = self.paraphrases_dir / f"{ds_name}_questions.jsonl"
            save_jsonl(questions, path)
            logger.info(f"Saved {len(questions)} {ds_name} questions to {path}")
            
            for q in questions:
                all_questions[q.id] = q
        
        return all_questions
    
    def stage2_generate_paraphrases(
        self,
        questions: dict,
        num_paraphrases: int = 4,
        use_mock: bool = False
    ) -> dict:
        """Stage 2: Generate paraphrases for all questions."""
        logger.info("="*50)
        logger.info("STAGE 2: Generating paraphrases")
        logger.info("="*50)
        
        generator = ParaphraseGenerator(use_mock=use_mock)
        
        # Group questions by dataset
        by_dataset = {}
        for q in questions.values():
            ds = q.source_dataset
            if ds not in by_dataset:
                by_dataset[ds] = []
            by_dataset[ds].append(q)
        
        all_paraphrases = {}
        
        for ds_name, ds_questions in by_dataset.items():
            logger.info(f"Generating paraphrases for {ds_name}...")
            
            paraphrases = generator.generate_paraphrases_batch(
                ds_questions,
                num_paraphrases_per_question=num_paraphrases,
                show_progress=True
            )
            
            # Save
            flat_paraphrases = []
            for q_id, p_list in paraphrases.items():
                all_paraphrases[q_id] = p_list
                flat_paraphrases.extend(p_list)
            
            path = self.paraphrases_dir / f"{ds_name}_paraphrases.jsonl"
            save_jsonl(flat_paraphrases, path)
            logger.info(f"Saved {len(flat_paraphrases)} paraphrases to {path}")
        
        return all_paraphrases
    
    def load_existing_explanations(self, model: str) -> dict:
        """Load existing explanations for a model to enable resume."""
        existing = {"original": {}, "paraphrase": {}}
        model_safe = model.replace("/", "_")
        
        # Check for original explanations
        for ef in self.explanations_dir.glob(f"{model_safe}*_original.jsonl"):
            logger.info(f"Loading existing original explanations from {ef}")
            try:
                for e in load_jsonl(ef, Explanation):
                    existing["original"][e.question_id] = e
            except Exception as ex:
                logger.warning(f"Error loading {ef}: {ex}. Skipping this file.")
        
        # Check for paraphrase explanations
        for ef in self.explanations_dir.glob(f"{model_safe}*_paraphrase.jsonl"):
            logger.info(f"Loading existing paraphrase explanations from {ef}")
            try:
                for e in load_jsonl(ef, Explanation):
                    existing["paraphrase"][e.question_id] = e
            except Exception as ex:
                logger.warning(f"Error loading {ef}: {ex}. Skipping this file.")
        
        logger.info(f"Found {len(existing['original'])} existing original explanations")
        logger.info(f"Found {len(existing['paraphrase'])} existing paraphrase explanations")
        
        return existing
    
    def stage3_generate_explanations(
        self,
        questions: dict,
        paraphrases: dict,
        models: list,
        prompt_types: list = None,
        temperatures: list = None,
        use_mock: bool = False,
        quantize: bool = False,
        resume: bool = False
    ) -> dict:
        """Stage 3: Generate CoT explanations with resume support."""
        logger.info("="*50)
        logger.info("STAGE 3: Generating explanations")
        logger.info("="*50)
        
        if prompt_types is None:
            prompt_types = ["zero_shot_cot"]
        if temperatures is None:
            temperatures = [0.0]
        
        all_explanations = {
            "original": {},
            "paraphrase": {}
        }
        
        for model in models:
            logger.info(f"Setting up model: {model}")
            model_safe = model.replace("/", "_")
            
            # Load existing explanations if resuming
            existing = {"original": {}, "paraphrase": {}}
            if resume:
                existing = self.load_existing_explanations(model)
                all_explanations["original"].update(existing["original"])
                all_explanations["paraphrase"].update(existing["paraphrase"])
            
            # Create generator for this model
            generator = ExplanationGenerator(
                model_name=model,
                device=self.device,
                load_in_4bit=quantize,
                use_mock=use_mock
            )
            
            for prompt_type in prompt_types:
                for temp in temperatures:
                    config = GenerationConfig(
                        prompt_type=prompt_type,
                        temperature=temp
                    )
                    
                    logger.info(f"  Generating with {prompt_type}, T={temp}")
                    
                    # Filter out already-generated original questions
                    question_list = [q for q in questions.values() 
                                    if q.id not in existing["original"]]
                    
                    if question_list:
                        logger.info(f"  Generating {len(question_list)} original explanations (skipping {len(existing['original'])} existing)")
                        original_exps = generator.generate_explanations_batch(
                            question_list, config=config, show_progress=True
                        )
                        
                        for exp in original_exps:
                            all_explanations["original"][exp.question_id] = exp
                    else:
                        logger.info(f"  All {len(existing['original'])} original explanations already exist, skipping")
                        original_exps = list(existing["original"].values())
                    
                    # Save all originals (existing + new)
                    all_original = list(all_explanations["original"].values())
                    path = self.explanations_dir / f"{model_safe}_{prompt_type}_T{temp}_original.jsonl"
                    save_jsonl(all_original, path)
                    
                    # Filter out already-generated paraphrase explanations
                    para_list = []
                    for p_list in paraphrases.values():
                        for p in p_list:
                            if p.id not in existing["paraphrase"]:
                                para_list.append(p)
                    
                    total_paraphrases = sum(len(p_list) for p_list in paraphrases.values())
                    
                    if para_list:
                        logger.info(f"  Generating {len(para_list)} paraphrase explanations (skipping {len(existing['paraphrase'])} existing)")
                        para_exps = generator.generate_explanations_batch(
                            para_list,
                            original_questions=questions,
                            config=config,
                            show_progress=True
                        )
                        
                        for exp in para_exps:
                            all_explanations["paraphrase"][exp.question_id] = exp
                    else:
                        logger.info(f"  All {len(existing['paraphrase'])} paraphrase explanations already exist, skipping")
                        para_exps = list(existing["paraphrase"].values())
                    
                    # Save all paraphrases (existing + new)
                    all_para = list(all_explanations["paraphrase"].values())
                    path = self.explanations_dir / f"{model_safe}_{prompt_type}_T{temp}_paraphrase.jsonl"
                    save_jsonl(all_para, path)
                    
                    logger.info(f"    Total: {len(all_explanations['original'])} original, {len(all_explanations['paraphrase'])} paraphrase explanations")
            
            # Clean up model to free memory
            del generator
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_explanations
    
    def stage4_evaluate(
        self,
        questions: dict,
        paraphrases: dict,
        explanations: dict,
        models: list,
        use_llm_judge: bool = False
    ) -> dict:
        """Stage 4: Evaluate explanation consistency."""
        logger.info("="*50)
        logger.info("STAGE 4: Evaluating consistency")
        logger.info("="*50)
        
        config = EvaluationConfig(
            models=models,
            use_llm_judge=use_llm_judge,
            device=self.device,
            output_dir=str(self.results_dir)
        )
        
        pipeline = EvaluationPipeline(config)
        
        # Create pairs
        pairs = pipeline.create_explanation_pairs(
            questions,
            paraphrases,
            explanations["original"],
            explanations["paraphrase"]
        )
        logger.info(f"Created {len(pairs)} explanation pairs")
        
        # Evaluate
        pairs = pipeline.evaluate_pairs(pairs)
        
        # Get results for each model
        results = {}
        for model in models:
            model_safe = model.replace("/", "_")
            model_pairs = [p for p in pairs if model_safe in p.explanation_original.model.replace("/", "_")]
            
            if model_pairs:
                result = pipeline.compute_aggregate_results(
                    model_pairs, model, "zero_shot_cot", 0.0
                )
                analysis = pipeline.analyze_divergence_patterns(model_pairs)
                
                results[model] = {
                    "result": result,
                    "analysis": analysis,
                    "pairs": model_pairs
                }
                
                pipeline.save_results(model_pairs, result, suffix=f"_{model_safe}")
        
        return results
    
    def stage5_report(self, results: dict):
        """Stage 5: Generate final report."""
        logger.info("="*50)
        logger.info("STAGE 5: Generating report")
        logger.info("="*50)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "summary": {}
        }
        
        for model, data in results.items():
            result = data["result"]
            analysis = data["analysis"]
            
            report["models"][model] = {
                "expara_sim": result.avg_expara_sim,
                "expara_sim_std": result.std_expara_sim,
                "answer_consistency": result.answer_consistency_rate,
                "metrics": {
                    "semantic_sim": result.avg_semantic_sim,
                    "entailment_sym": result.avg_entailment_sym,
                    "step_align": result.avg_step_align,
                    "fact_overlap": result.avg_fact_overlap,
                    "llm_judge": result.avg_llm_judge
                },
                "divergence_rate": analysis["divergence_rate"],
                "by_task_type": result.by_task_type,
                "by_paraphrase_type": result.by_paraphrase_type
            }
        
        # Compute summary statistics
        if results:
            scores = [r["result"].avg_expara_sim for r in results.values()]
            report["summary"] = {
                "num_models": len(results),
                "avg_expara_sim": sum(scores) / len(scores),
                "best_model": max(results.keys(), key=lambda m: results[m]["result"].avg_expara_sim),
                "worst_model": min(results.keys(), key=lambda m: results[m]["result"].avg_expara_sim)
            }
        
        # Save report
        report_path = self.results_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved final report to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPARA BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for model, data in sorted(results.items(), 
                                   key=lambda x: x[1]["result"].avg_expara_sim, 
                                   reverse=True):
            r = data["result"]
            print(f"\n{model}:")
            print(f"  ExPara-Sim: {r.avg_expara_sim:.4f} (±{r.std_expara_sim:.4f})")
            print(f"  Answer Consistency: {r.answer_consistency_rate:.2%}")
            print(f"  Divergence Rate: {data['analysis']['divergence_rate']:.2%}")
        
        print("\n" + "="*60)
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Run complete ExPara pipeline")
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-7B-Instruct"],
        help="Models to evaluate (HuggingFace model names)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--num_paraphrases",
        type=int,
        default=4,
        help="Number of paraphrases per question"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock clients (for testing)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (saves memory)"
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip data generation, use existing files"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing explanations (skip already generated)"
    )
    parser.add_argument(
        "--skip_paraphrases",
        action="store_true",
        help="Skip paraphrase generation, load from existing files"
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Include LLM judge metric"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for models"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ExParaPipeline(
        output_dir=args.output,
        device=args.device
    )
    
    if args.skip_generation:
        # Load existing data
        logger.info("Loading existing data...")
        
        questions = {}
        paraphrases = {}
        explanations = {"original": {}, "paraphrase": {}}
        
        for qf in pipeline.paraphrases_dir.glob("*_questions.jsonl"):
            for q in load_jsonl(qf, Question):
                questions[q.id] = q
        
        for pf in pipeline.paraphrases_dir.glob("*_paraphrases.jsonl"):
            for p in load_jsonl(pf, Paraphrase):
                if p.original_question_id not in paraphrases:
                    paraphrases[p.original_question_id] = []
                paraphrases[p.original_question_id].append(p)
        
        for ef in pipeline.explanations_dir.glob("*_original.jsonl"):
            for e in load_jsonl(ef, Explanation):
                explanations["original"][e.question_id] = e
        
        for ef in pipeline.explanations_dir.glob("*_paraphrase.jsonl"):
            for e in load_jsonl(ef, Explanation):
                explanations["paraphrase"][e.question_id] = e
    else:
        # Check if we should load existing questions/paraphrases
        if args.resume or args.skip_paraphrases:
            logger.info("Loading existing questions and paraphrases...")
            questions = {}
            paraphrases = {}
            
            for qf in pipeline.paraphrases_dir.glob("*_questions.jsonl"):
                for q in load_jsonl(qf, Question):
                    questions[q.id] = q
            
            for pf in pipeline.paraphrases_dir.glob("*_paraphrases.jsonl"):
                for p in load_jsonl(pf, Paraphrase):
                    if p.original_question_id not in paraphrases:
                        paraphrases[p.original_question_id] = []
                    paraphrases[p.original_question_id].append(p)
            
            logger.info(f"Loaded {len(questions)} questions and {sum(len(v) for v in paraphrases.values())} paraphrases")
            
            # If we need fewer samples, subsample
            if len(questions) > args.num_samples * 4:  # Rough estimate: 4 datasets
                logger.info(f"Subsampling to {args.num_samples} per dataset...")
                # Group by dataset and subsample
                by_dataset = {}
                for q in questions.values():
                    ds = q.source_dataset
                    if ds not in by_dataset:
                        by_dataset[ds] = []
                    by_dataset[ds].append(q)
                
                new_questions = {}
                new_paraphrases = {}
                for ds, qs in by_dataset.items():
                    for q in qs[:args.num_samples]:
                        new_questions[q.id] = q
                        if q.id in paraphrases:
                            new_paraphrases[q.id] = paraphrases[q.id]
                
                questions = new_questions
                paraphrases = new_paraphrases
                logger.info(f"Subsampled to {len(questions)} questions")
        else:
            # Generate fresh
            questions = pipeline.stage1_load_datasets(
                args.datasets, args.num_samples
            )
            
            paraphrases = pipeline.stage2_generate_paraphrases(
                questions, args.num_paraphrases, args.use_mock
            )
        
        # Generate explanations with resume support
        explanations = pipeline.stage3_generate_explanations(
            questions, paraphrases, args.models,
            use_mock=args.use_mock,
            quantize=args.quantize,
            resume=args.resume
        )
    
    # Evaluate
    results = pipeline.stage4_evaluate(
        questions, paraphrases, explanations,
        args.models if not args.use_mock else ["mock"],
        use_llm_judge=args.use_llm_judge
    )
    
    # Generate report
    pipeline.stage5_report(results)
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()