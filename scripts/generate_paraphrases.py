#!/usr/bin/env python3
"""
Generate paraphrases using open-source models.
NO API KEYS NEEDED!

Usage:
    python scripts/generate_paraphrases.py --dataset gsm8k --num_samples 100
    python scripts/generate_paraphrases.py --model Qwen/Qwen2.5-3B-Instruct --quantize
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import DatasetLoader
from data.paraphrase_generator import ParaphraseGenerator, ParaphraseValidator
from data.models import save_jsonl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrases using open-source models (no API keys needed)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "strategyqa", "arc_challenge", "logiqa", "all"],
        default="all",
        help="Dataset to process"
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
        default="data/paraphrases/",
        help="Output directory"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model for paraphrase generation"
    )
    parser.add_argument(
        "--use_t5",
        action="store_true",
        help="Use T5 paraphrase model instead (faster, simpler)"
    )
    parser.add_argument(
        "--t5_model",
        type=str,
        default="Vamsi/T5_Paraphrase_Paws",
        help="T5 model for paraphrasing"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 4-bit quantization (saves memory)"
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
    
    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate paraphrases with NLI model"
    )
    parser.add_argument(
        "--nli_threshold",
        type=float,
        default=0.5,
        help="NLI threshold for validation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    loader = DatasetLoader()
    
    if args.dataset == "all":
        questions_dict = loader.load_all(args.num_samples)
    else:
        load_func = getattr(loader, f"load_{args.dataset}")
        questions_dict = {args.dataset: load_func(args.num_samples)}
    
    # Initialize generator
    logger.info(f"Initializing paraphrase generator...")
    if args.use_mock:
        generator = ParaphraseGenerator(use_mock=True)
    elif args.use_t5:
        generator = ParaphraseGenerator(
            use_t5=True,
            t5_model=args.t5_model,
            device=args.device
        )
    else:
        generator = ParaphraseGenerator(
            model_name=args.model,
            device=args.device,
            load_in_4bit=args.quantize
        )
    
    # Initialize validator if needed
    validator = None
    if args.validate and not args.use_mock:
        logger.info("Initializing paraphrase validator...")
        validator = ParaphraseValidator(device=args.device)
    
    # Process each dataset
    for dataset_name, questions in questions_dict.items():
        logger.info(f"Processing {dataset_name} ({len(questions)} questions)...")
        
        # Generate paraphrases
        paraphrases = generator.generate_paraphrases_batch(
            questions,
            num_paraphrases_per_question=args.num_paraphrases,
            show_progress=True
        )
        
        # Validate if requested
        if validator:
            logger.info("Validating paraphrases...")
            questions_by_id = {q.id: q for q in questions}
            for q_id, p_list in paraphrases.items():
                original = questions_by_id.get(q_id)
                if original:
                    for p in p_list:
                        validator.validate_paraphrase(original, p, args.nli_threshold)
        
        # Save questions
        questions_path = os.path.join(args.output, f"{dataset_name}_questions.jsonl")
        save_jsonl(questions, questions_path)
        logger.info(f"Saved {len(questions)} questions to {questions_path}")
        
        # Save paraphrases
        all_paraphrases = []
        for p_list in paraphrases.values():
            all_paraphrases.extend(p_list)
        
        paraphrases_path = os.path.join(args.output, f"{dataset_name}_paraphrases.jsonl")
        save_jsonl(all_paraphrases, paraphrases_path)
        logger.info(f"Saved {len(all_paraphrases)} paraphrases to {paraphrases_path}")
        
        # Statistics
        valid_count = sum(1 for p in all_paraphrases if p.is_valid)
        logger.info(f"  Valid paraphrases: {valid_count}/{len(all_paraphrases)}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
