#!/usr/bin/env python3
"""
Create CLEAN human evaluation dataset for ExPara divergent pairs.
FIXED VERSION - Correctly maps original questions to paraphrases.

Usage:
    pip install sentence-transformers
    python scripts/create_clean_human_eval_v2.py --data_dir data/ --output human_eval_clean/ --num_pairs 100
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("WARNING: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    print("Falling back to simple word overlap similarity.\n")


class SemanticSimilarityChecker:
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        if self.use_embeddings:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded!\n")
        else:
            self.model = None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        if self.use_embeddings:
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)
            return util.cos_sim(emb1, emb2).item()
        else:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0


def load_original_questions(paraphrases_dir: Path) -> dict:
    """Load original questions: question_id -> question_text"""
    questions = {}
    for qf in paraphrases_dir.glob("*_questions.jsonl"):
        with open(qf, encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                questions[q['id']] = q['text']
    return questions


def load_paraphrases(paraphrases_dir: Path) -> dict:
    """Load paraphrases: paraphrase_id -> {original_id, text, type}"""
    paraphrases = {}
    for pf in paraphrases_dir.glob("*_paraphrases.jsonl"):
        with open(pf, encoding='utf-8') as f:
            for line in f:
                p = json.loads(line)
                paraphrases[p['id']] = {
                    'original_id': p['original_question_id'],
                    'text': p['text'],
                    'type': p.get('paraphrase_type', 'unknown')
                }
    return paraphrases


def load_original_explanations(explanations_dir: Path) -> dict:
    """Load original explanations: (model, question_id) -> explanation"""
    explanations = {}
    for ef in explanations_dir.glob("*_original.jsonl"):
        with open(ef, encoding='utf-8') as f:
            for line in f:
                exp = json.loads(line)
                model = exp['model'].replace('/', '_')
                key = (model, exp['question_id'])
                explanations[key] = exp
    return explanations


def load_paraphrase_explanations(explanations_dir: Path) -> dict:
    """Load paraphrase explanations: (model, paraphrase_id) -> explanation"""
    explanations = {}
    for ef in explanations_dir.glob("*_paraphrase.jsonl"):
        with open(ef, encoding='utf-8') as f:
            for line in f:
                exp = json.loads(line)
                model = exp['model'].replace('/', '_')
                # question_id in paraphrase explanation IS the paraphrase_id
                key = (model, exp['question_id'])
                explanations[key] = exp
    return explanations


def answers_match(answer1: str, answer2: str) -> bool:
    """Check if two answers are essentially the same."""
    a1 = answer1.strip().lower()[:50]
    a2 = answer2.strip().lower()[:50]
    
    if a1 == a2:
        return True
    if a1 in a2 or a2 in a1:
        return True
    
    import re
    nums1 = set(re.findall(r'\d+\.?\d*', a1))
    nums2 = set(re.findall(r'\d+\.?\d*', a2))
    if nums1 and nums2 and nums1 == nums2:
        return True
    
    return False


def explanations_diverge(exp1: str, exp2: str) -> bool:
    """Check if two explanations are meaningfully different."""
    len_diff = abs(len(exp1) - len(exp2))
    if len_diff > 200:
        return True
    if exp1[:100] != exp2[:100]:
        return True
    return False


def find_divergent_pairs(
    original_questions: dict,
    paraphrases: dict,
    original_explanations: dict,
    paraphrase_explanations: dict,
    similarity_checker,
    min_similarity: float = 0.6,
    max_pairs: int = 100
):
    """Find pairs with same answer but different explanations."""
    
    candidates = []
    stats = {
        'total_examined': 0,
        'no_original_explanation': 0,
        'low_similarity': 0,
        'different_answer': 0,
        'same_explanation': 0,
        'passed': 0
    }
    
    print("Processing pairs...")
    
    # Iterate through paraphrase explanations
    for (model, para_id), para_exp in paraphrase_explanations.items():
        
        # Get paraphrase info
        if para_id not in paraphrases:
            continue
        
        para_info = paraphrases[para_id]
        original_id = para_info['original_id']
        para_text = para_info['text']
        para_type = para_info['type']
        
        # Get original question text
        if original_id not in original_questions:
            continue
        
        original_text = original_questions[original_id]
        
        # Get original explanation for this model
        orig_key = (model, original_id)
        if orig_key not in original_explanations:
            stats['no_original_explanation'] += 1
            continue
        
        orig_exp = original_explanations[orig_key]
        
        stats['total_examined'] += 1
        
        # STEP 1: Check question similarity (FILTER BAD PARAPHRASES)
        similarity = similarity_checker.compute_similarity(original_text, para_text)
        
        if similarity < min_similarity:
            stats['low_similarity'] += 1
            continue
        
        # STEP 2: Check same answer
        if not answers_match(orig_exp['final_answer'], para_exp['final_answer']):
            stats['different_answer'] += 1
            continue
        
        # STEP 3: Check different explanation
        if not explanations_diverge(orig_exp['reasoning'], para_exp['reasoning']):
            stats['same_explanation'] += 1
            continue
        
        stats['passed'] += 1
        
        candidates.append({
            'model': model,
            'original_question': original_text,
            'paraphrased_question': para_text,
            'question_similarity': round(similarity, 3),
            'paraphrase_type': para_type,
            'original_answer': orig_exp['final_answer'],
            'paraphrase_answer': para_exp['final_answer'],
            'original_reasoning': orig_exp['reasoning'],
            'paraphrase_reasoning': para_exp['reasoning'],
            'original_question_id': original_id,
            'paraphrase_id': para_id
        })
    
    print(f"\n=== FILTERING STATISTICS ===")
    print(f"Total pairs examined: {stats['total_examined']}")
    print(f"No original explanation found: {stats['no_original_explanation']}")
    print(f"Filtered (low similarity < {min_similarity}): {stats['low_similarity']}")
    print(f"Filtered (different answers): {stats['different_answer']}")
    print(f"Filtered (same explanation): {stats['same_explanation']}")
    print(f"PASSED all filters: {stats['passed']}")
    
    # Sample if needed
    if len(candidates) > max_pairs:
        random.seed(42)
        candidates = random.sample(candidates, max_pairs)
    
    # Sort by similarity
    candidates.sort(key=lambda x: x['question_similarity'], reverse=True)
    
    return candidates, stats


def create_output_files(pairs: list, output_dir: Path, stats: dict):
    """Create all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Instructions
    instructions = """
================================================================================
HUMAN EVALUATION INSTRUCTIONS - ExPara Study (FIXED VERSION)
================================================================================

These pairs have been VERIFIED:
✓ Original and paraphrased questions are semantically similar (>60%)
✓ Both received the SAME final answer
✓ Explanations appear different by automated metrics

YOUR TASK:
Rate how similar the two EXPLANATIONS are (not the questions).

RATING SCALE:
1 = Completely Different reasoning
2 = Mostly Different (some overlap)
3 = Somewhat Similar (same approach, different details)
4 = Mostly Similar (same reasoning, minor wording differences)
5 = Essentially Identical

FOCUS ON:
- Do they use the SAME logical steps?
- Do they cite the SAME facts/numbers?
- Would the same expert give BOTH explanations?

================================================================================
"""
    
    with open(output_dir / "INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    # JSON data
    with open(output_dir / "eval_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    # Evaluation form
    eval_doc = f"""
================================================================================
EXPARA HUMAN EVALUATION - {len(pairs)} PAIRS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

Stats: {stats['total_examined']} examined, {stats['low_similarity']} bad paraphrases filtered

"""
    
    for i, pair in enumerate(pairs, 1):
        eval_doc += f"""
{'='*80}
PAIR {i} | Model: {pair['model']} | Similarity: {pair['question_similarity']:.0%} ✓
{'='*80}

ORIGINAL QUESTION:
{pair['original_question']}

PARAPHRASED QUESTION:
{pair['paraphrased_question']}

ANSWER: {pair['original_answer'][:200]}{'...' if len(pair['original_answer']) > 200 else ''}

--- ORIGINAL EXPLANATION ---
{pair['original_reasoning'][:1200]}{'...' if len(pair['original_reasoning']) > 1200 else ''}

--- PARAPHRASE EXPLANATION ---
{pair['paraphrase_reasoning'][:1200]}{'...' if len(pair['paraphrase_reasoning']) > 1200 else ''}

YOUR RATING (1-5): _____   NOTES: _______________________

"""
    
    with open(output_dir / "EVALUATION_FORM.txt", 'w', encoding='utf-8') as f:
        f.write(eval_doc)
    
    # CSV
    csv = "pair_id,model,similarity,rating,notes\n"
    for i, pair in enumerate(pairs, 1):
        csv += f'{i},"{pair["model"]}",{pair["question_similarity"]},,\n'
    
    with open(output_dir / "ratings.csv", 'w', encoding='utf-8') as f:
        f.write(csv)
    
    print(f"\n✅ Created files in {output_dir}/")
    print(f"   - INSTRUCTIONS.txt")
    print(f"   - EVALUATION_FORM.txt ({len(pairs)} pairs)")
    print(f"   - eval_pairs.json")
    print(f"   - ratings.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default="human_eval_clean")
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--min_similarity", type=float, default=0.6)
    parser.add_argument("--no_embeddings", action="store_true")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Initialize similarity checker
    similarity_checker = SemanticSimilarityChecker(use_embeddings=not args.no_embeddings)
    
    # Load all data
    print("Loading original questions...")
    original_questions = load_original_questions(data_dir / "paraphrases")
    print(f"  Loaded {len(original_questions)} questions")
    
    print("Loading paraphrases...")
    paraphrases = load_paraphrases(data_dir / "paraphrases")
    print(f"  Loaded {len(paraphrases)} paraphrases")
    
    print("Loading original explanations...")
    original_explanations = load_original_explanations(data_dir / "explanations")
    print(f"  Loaded {len(original_explanations)} explanations")
    
    print("Loading paraphrase explanations...")
    paraphrase_explanations = load_paraphrase_explanations(data_dir / "explanations")
    print(f"  Loaded {len(paraphrase_explanations)} explanations")
    
    # Find divergent pairs
    pairs, stats = find_divergent_pairs(
        original_questions,
        paraphrases,
        original_explanations,
        paraphrase_explanations,
        similarity_checker,
        min_similarity=args.min_similarity,
        max_pairs=args.num_pairs
    )
    
    # Create output
    create_output_files(pairs, Path(args.output), stats)
    
    # Show sample
    if pairs:
        print(f"\n=== SAMPLE PAIR ===")
        p = pairs[0]
        print(f"Similarity: {p['question_similarity']:.0%}")
        print(f"Original Q: {p['original_question'][:60]}...")
        print(f"Paraphrase Q: {p['paraphrased_question'][:60]}...")


if __name__ == "__main__":
    main()