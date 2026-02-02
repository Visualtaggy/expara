#!/usr/bin/env python3
"""
Generate chain-of-thought explanations using open-source models.
NO API KEYS NEEDED!

Usage:
    python scripts/generate_explanations.py --model Qwen/Qwen2.5-7B-Instruct --input data/paraphrases/
    python scripts/generate_explanations.py --model Qwen/Qwen2.5-7B-Instruct --quantize --input data/paraphrases/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.models import Question, Paraphrase, load_jsonl, save_jsonl
from evaluation.explanation_generator import ExplanationGenerator, GenerationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Recommended models
RECOMMENDED_MODELS = {
    "small": "google/gemma-2-2b-it",
    "medium": "Qwen/Qwen2.5-3B-Instruct",
    "large": "Qwen/Qwen2.5-7B-Instruct",
    "xlarge": "Qwen/Qwen2.5-14B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT explanations using open-source models (no API keys needed)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name (or: small, medium, large, xlarge)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with questions and paraphrases"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/explanations/",
        help="Output directory"
    )
    
    # Generation config
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["zero_shot_cot", "few_shot_cot"],
        default="zero_shot_cot",
        help="Prompting strategy"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0 for deterministic)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    
    # Model options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (saves ~75%% memory)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)"
    )
    
    # Testing
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock generator for testing"
    )
    
    # Dataset filter
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Resolve model name shorthand
    if args.model in RECOMMENDED_MODELS:
        args.model = RECOMMENDED_MODELS[args.model]
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize generator
    logger.info(f"Initializing explanation generator with {args.model}...")
    
    generator = ExplanationGenerator(
        model_name=args.model,
        device=args.device,
        load_in_4bit=args.quantize,
        use_mock=args.use_mock
    )
    
    config = GenerationConfig(
        prompt_type=args.prompt_type,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )
    
    # Find input files
    input_path = Path(args.input)
    question_files = list(input_path.glob("*_questions.jsonl"))
    
    if args.dataset:
        question_files = [f for f in question_files if args.dataset in f.name]
    
    if not question_files:
        logger.error(f"No question files found in {args.input}")
        return
    
    # Process each dataset
    for questions_path in question_files:
        dataset_name = questions_path.stem.replace("_questions", "")
        logger.info(f"Processing {dataset_name}...")
        
        # Load questions
        questions = load_jsonl(questions_path, Question)
        questions_by_id = {q.id: q for q in questions}
        logger.info(f"  Loaded {len(questions)} questions")
        
        # Load paraphrases
        paraphrases_path = questions_path.parent / f"{dataset_name}_paraphrases.jsonl"
        if paraphrases_path.exists():
            paraphrases = load_jsonl(paraphrases_path, Paraphrase)
            logger.info(f"  Loaded {len(paraphrases)} paraphrases")
        else:
            paraphrases = []
            logger.warning(f"  No paraphrases file found")
        
        # Generate explanations for original questions
        logger.info("  Generating explanations for original questions...")
        original_explanations = generator.generate_explanations_batch(
            questions,
            config=config,
            show_progress=True
        )
        
        # Save original explanations
        model_short = args.model.split("/")[-1]
        output_path = Path(args.output) / f"{dataset_name}_{model_short}_original_explanations.jsonl"
        save_jsonl(original_explanations, output_path)
        logger.info(f"  Saved {len(original_explanations)} original explanations to {output_path}")
        
        # Generate explanations for paraphrases
        if paraphrases:
            logger.info("  Generating explanations for paraphrases...")
            paraphrase_explanations = generator.generate_explanations_batch(
                paraphrases,
                original_questions=questions_by_id,
                config=config,
                show_progress=True
            )
            
            # Save paraphrase explanations
            output_path = Path(args.output) / f"{dataset_name}_{model_short}_paraphrase_explanations.jsonl"
            save_jsonl(paraphrase_explanations, output_path)
            logger.info(f"  Saved {len(paraphrase_explanations)} paraphrase explanations to {output_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
