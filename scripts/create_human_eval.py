#!/usr/bin/env python3
"""
Create CLEAN human evaluation dataset for ExPara divergent pairs.

This script:
1. Loads all explanation pairs
2. FILTERS OUT bad paraphrases (low semantic similarity between questions)
3. Extracts divergent pairs (same answer, different explanation)
4. Formats them for human annotation

Usage:
    python scripts/create_clean_human_eval.py --data_dir data/ --output human_eval_clean/ --num_pairs 100
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
    print("WARNING: sentence-transformers not installed. Install with: pip install sentence-transformers")
    print("Falling back to simple word overlap similarity.\n")


class SemanticSimilarityChecker:
    """Check if two questions are semantically similar."""
    
    def __init__(self, use_embeddings=True):
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        if self.use_embeddings:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded!\n")
        else:
            self.model = None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts (0-1)."""
        if self.use_embeddings:
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            return similarity
        else:
            # Fallback: word overlap (Jaccard similarity)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0


def load_explanations(explanations_dir: Path, model_pattern: str = ""):
    """Load all explanations from jsonl files."""
    originals = {}
    paraphrases = {}
    
    if model_pattern and model_pattern != "*":
        orig_pattern = f"{model_pattern}*_original.jsonl"
        para_pattern = f"{model_pattern}*_paraphrase.jsonl"
    else:
        orig_pattern = "*_original.jsonl"
        para_pattern = "*_paraphrase.jsonl"
    
    for ef in explanations_dir.glob(orig_pattern):
        with open(ef, encoding='utf-8') as f:
            for line in f:
                exp = json.loads(line)
                key = (exp['model'].replace('/', '_'), exp['question_id'])
                originals[key] = exp
    
    for ef in explanations_dir.glob(para_pattern):
        with open(ef, encoding='utf-8') as f:
            for line in f:
                exp = json.loads(line)
                key = (exp['model'].replace('/', '_'), exp['question_id'])
                paraphrases[key] = exp
    
    return originals, paraphrases


def load_paraphrase_mappings(paraphrases_dir: Path):
    """Load paraphrase metadata to map paraphrase_id -> original_question_id."""
    mappings = {}
    
    for pf in paraphrases_dir.glob("*_paraphrases.jsonl"):
        with open(pf, encoding='utf-8') as f:
            for line in f:
                p = json.loads(line)
                mappings[p['id']] = {
                    'original_id': p['original_question_id'],
                    'para_text': p['text'],
                    'para_type': p.get('paraphrase_type', 'unknown')
                }
    
    return mappings


def load_original_questions(paraphrases_dir: Path):
    """Load original questions to get their text."""
    questions = {}
    
    for qf in paraphrases_dir.glob("*_questions.jsonl"):
        with open(qf, encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                questions[q['id']] = q['text']
    
    return questions


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
    
    headers1 = exp1.count('###') + exp1.count('**Step')
    headers2 = exp2.count('###') + exp2.count('**Step')
    if abs(headers1 - headers2) > 2:
        return True
    
    return False


def find_clean_divergent_pairs(originals, paraphrases, para_mappings, original_questions,
                                similarity_checker, min_similarity=0.6, max_pairs=100):
    """Find pairs with same answer but different explanations, filtering bad paraphrases."""
    
    candidates = []
    stats = {
        'total_pairs': 0,
        'filtered_low_similarity': 0,
        'filtered_different_answer': 0,
        'filtered_same_explanation': 0,
        'passed': 0
    }
    
    print("Processing pairs...")
    
    for (model, para_id), para_exp in paraphrases.items():
        if para_id not in para_mappings:
            continue
        
        orig_id = para_mappings[para_id]['original_id']
        orig_key = (model, orig_id)
        
        if orig_key not in originals:
            continue
        
        if orig_id not in original_questions:
            continue
        
        stats['total_pairs'] += 1
        
        orig_exp = originals[orig_key]
        orig_question = original_questions[orig_id]
        para_question = para_mappings[para_id]['para_text']
        
        # STEP 1: Check question similarity (FILTER BAD PARAPHRASES)
        similarity = similarity_checker.compute_similarity(orig_question, para_question)
        
        if similarity < min_similarity:
            stats['filtered_low_similarity'] += 1
            continue
        
        # STEP 2: Check same answer
        if not answers_match(orig_exp['final_answer'], para_exp['final_answer']):
            stats['filtered_different_answer'] += 1
            continue
        
        # STEP 3: Check different explanation
        if not explanations_diverge(orig_exp['reasoning'], para_exp['reasoning']):
            stats['filtered_same_explanation'] += 1
            continue
        
        stats['passed'] += 1
        
        candidates.append({
            'model': model,
            'original_question': orig_question,
            'paraphrased_question': para_question,
            'question_similarity': round(similarity, 3),
            'paraphrase_type': para_mappings[para_id]['para_type'],
            'original_answer': orig_exp['final_answer'],
            'paraphrase_answer': para_exp['final_answer'],
            'original_reasoning': orig_exp['reasoning'],
            'paraphrase_reasoning': para_exp['reasoning'],
            'original_question_id': orig_id,
            'paraphrase_id': para_id
        })
    
    print(f"\n=== FILTERING STATISTICS ===")
    print(f"Total pairs examined: {stats['total_pairs']}")
    print(f"Filtered (low question similarity < {min_similarity}): {stats['filtered_low_similarity']}")
    print(f"Filtered (different answers): {stats['filtered_different_answer']}")
    print(f"Filtered (same explanation): {stats['filtered_same_explanation']}")
    print(f"PASSED all filters: {stats['passed']}")
    
    # Sample if we have more than needed
    if len(candidates) > max_pairs:
        random.seed(42)
        candidates = random.sample(candidates, max_pairs)
    
    # Sort by similarity (highest first) to show best examples
    candidates.sort(key=lambda x: x['question_similarity'], reverse=True)
    
    return candidates, stats


def create_human_eval_document(divergent_pairs, output_dir: Path, stats: dict):
    """Create human evaluation documents."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = """
================================================================================
HUMAN EVALUATION INSTRUCTIONS FOR EXPARA STUDY (CLEAN VERSION)
================================================================================

IMPORTANT: These pairs have been PRE-FILTERED to ensure:
✓ Original and paraphrased questions are semantically similar (>60% similarity)
✓ Both questions received the SAME final answer
✓ The explanations appear different by automated metrics

YOUR TASK:
----------
For each pair, you will see:
1. An original question and its paraphrased version (VERIFIED as similar)
2. The model's answer to both (same answer)
3. The model's explanation/reasoning for both

You need to judge: Are the two explanations SEMANTICALLY EQUIVALENT?

RATING SCALE:
-------------
1 = Completely Different
    - Different reasoning approaches
    - Different facts or steps mentioned
    - Would confuse a reader if they saw both

2 = Mostly Different  
    - Some overlap but significant differences
    - Different structure or emphasis
    - Key reasoning steps differ

3 = Somewhat Similar
    - Same general approach but different details
    - Some steps match, others don't
    - Similar conclusion but different path

4 = Mostly Similar
    - Same reasoning with minor wording differences
    - Same key facts and steps
    - Structure might differ slightly

5 = Essentially Identical
    - Same reasoning, same facts, same structure
    - Only superficial wording differences
    - Clearly the same explanation

WHAT TO LOOK FOR:
-----------------
✓ Do both explanations use the SAME logical steps?
✓ Do they cite the SAME facts/numbers?
✓ Is the reasoning STRUCTURE similar?
✓ Would the same expert give BOTH explanations?

If explanations use different valid approaches → rate 2-3
If explanations have same approach, different wording → rate 4-5

================================================================================
"""
    
    with open(output_dir / "INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    # Save as JSON
    with open(output_dir / "eval_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(divergent_pairs, f, indent=2, ensure_ascii=False)
    
    # Create readable evaluation document
    eval_doc = f"""
================================================================================
EXPARA HUMAN EVALUATION - CLEAN VERSION - {len(divergent_pairs)} PAIRS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

FILTERING APPLIED:
- Question similarity threshold: >= 0.6
- Total pairs examined: {stats['total_pairs']}
- Filtered for low similarity: {stats['filtered_low_similarity']}
- Pairs with verified similar questions: {stats['passed']}

Please read INSTRUCTIONS.txt first!

For each pair, write your rating (1-5) in the space provided.

"""
    
    for i, pair in enumerate(divergent_pairs, 1):
        eval_doc += f"""
{'='*80}
PAIR {i} of {len(divergent_pairs)}
Model: {pair['model']}
Question Similarity: {pair['question_similarity']:.1%} ✓ VERIFIED SIMILAR
{'='*80}

ORIGINAL QUESTION:
{pair['original_question']}

PARAPHRASED QUESTION:
{pair['paraphrased_question']}

ANSWER (same for both):
{pair['original_answer'][:300]}{'...' if len(pair['original_answer']) > 300 else ''}

--------------------------------------------------------------------------------
ORIGINAL EXPLANATION:
--------------------------------------------------------------------------------
{pair['original_reasoning'][:1500]}{'...' if len(pair['original_reasoning']) > 1500 else ''}

--------------------------------------------------------------------------------
PARAPHRASED EXPLANATION:
--------------------------------------------------------------------------------
{pair['paraphrase_reasoning'][:1500]}{'...' if len(pair['paraphrase_reasoning']) > 1500 else ''}

--------------------------------------------------------------------------------
YOUR RATING (1-5): _____

NOTES (optional): ____________________________________________________________

"""
    
    eval_doc += """
================================================================================
END OF EVALUATION
================================================================================

Thank you for completing this evaluation!

Summary of your ratings:
- Number of 1s (Completely Different): ___
- Number of 2s (Mostly Different): ___
- Number of 3s (Somewhat Similar): ___
- Number of 4s (Mostly Similar): ___
- Number of 5s (Essentially Identical): ___

Key question: What percentage of pairs showed GENUINELY different reasoning? ____%

Notes:
________________________________________________________________________
________________________________________________________________________
"""
    
    with open(output_dir / "EVALUATION_FORM.txt", 'w', encoding='utf-8') as f:
        f.write(eval_doc)
    
    # Create CSV for data entry
    csv_content = "pair_id,model,question_similarity,rating,notes\n"
    for i, pair in enumerate(divergent_pairs, 1):
        csv_content += f'{i},"{pair["model"]}",{pair["question_similarity"]},"",""\n'
    
    with open(output_dir / "ratings.csv", 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    # Create summary of filtering
    summary = f"""
================================================================================
FILTERING SUMMARY
================================================================================

Total pairs examined: {stats['total_pairs']}

FILTERED OUT:
- Low question similarity (<0.6): {stats['filtered_low_similarity']} ({100*stats['filtered_low_similarity']/max(stats['total_pairs'],1):.1f}%)
- Different answers: {stats['filtered_different_answer']} ({100*stats['filtered_different_answer']/max(stats['total_pairs'],1):.1f}%)
- Same explanation: {stats['filtered_same_explanation']} ({100*stats['filtered_same_explanation']/max(stats['total_pairs'],1):.1f}%)

PASSED ALL FILTERS: {stats['passed']} ({100*stats['passed']/max(stats['total_pairs'],1):.1f}%)

Selected for evaluation: {len(divergent_pairs)}

Question similarity distribution of selected pairs:
"""
    
    if divergent_pairs:
        sims = [p['question_similarity'] for p in divergent_pairs]
        summary += f"- Min: {min(sims):.3f}\n"
        summary += f"- Max: {max(sims):.3f}\n"
        summary += f"- Mean: {sum(sims)/len(sims):.3f}\n"
    
    with open(output_dir / "FILTERING_SUMMARY.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✅ Created human evaluation files in {output_dir}/")
    print(f"   - INSTRUCTIONS.txt: Read this first")
    print(f"   - EVALUATION_FORM.txt: Main evaluation document ({len(divergent_pairs)} pairs)")
    print(f"   - ratings.csv: Quick rating entry")
    print(f"   - eval_pairs.json: Raw data for analysis")
    print(f"   - FILTERING_SUMMARY.txt: Statistics on filtering")


def main():
    parser = argparse.ArgumentParser(description="Create CLEAN human evaluation dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="human_eval_clean", help="Output directory")
    parser.add_argument("--num_pairs", type=int, default=100, help="Number of pairs to evaluate")
    parser.add_argument("--min_similarity", type=float, default=0.6, help="Minimum question similarity (0-1)")
    parser.add_argument("--model", type=str, default="", help="Model filter pattern")
    parser.add_argument("--no_embeddings", action="store_true", help="Use word overlap instead of embeddings")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    explanations_dir = data_dir / "explanations"
    paraphrases_dir = data_dir / "paraphrases"
    output_dir = Path(args.output)
    
    # Initialize similarity checker
    use_embeddings = not args.no_embeddings
    similarity_checker = SemanticSimilarityChecker(use_embeddings=use_embeddings)
    
    print("Loading explanations...")
    originals, paraphrases = load_explanations(explanations_dir, args.model)
    print(f"  Loaded {len(originals)} original, {len(paraphrases)} paraphrase explanations")
    
    print("Loading paraphrase mappings...")
    para_mappings = load_paraphrase_mappings(paraphrases_dir)
    print(f"  Loaded {len(para_mappings)} mappings")
    
    print("Loading original questions...")
    original_questions = load_original_questions(paraphrases_dir)
    print(f"  Loaded {len(original_questions)} questions")
    
    print(f"\nFinding clean divergent pairs (similarity >= {args.min_similarity})...")
    divergent, stats = find_clean_divergent_pairs(
        originals, paraphrases, para_mappings, original_questions,
        similarity_checker, 
        min_similarity=args.min_similarity,
        max_pairs=args.num_pairs
    )
    print(f"\nSelected {len(divergent)} pairs for evaluation")
    
    print("\nCreating evaluation documents...")
    create_human_eval_document(divergent, output_dir, stats)
    
    # Print sample
    if divergent:
        print("\n=== SAMPLE PAIR ===")
        sample = divergent[0]
        print(f"Question similarity: {sample['question_similarity']:.1%}")
        print(f"Original: {sample['original_question'][:80]}...")
        print(f"Paraphrase: {sample['paraphrased_question'][:80]}...")


if __name__ == "__main__":
    main()