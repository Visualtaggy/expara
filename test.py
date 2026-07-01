import json
from pathlib import Path

# Load the generated pairs
with open("human_eval_clean/eval_pairs.json", encoding='utf-8') as f:
    pairs = json.load(f)

print(f"Total pairs: {len(pairs)}\n")

# Check first 5 pairs - questions should be RELATED
print("=== CHECKING QUESTION PAIRS ===\n")
for i, p in enumerate(pairs[:5]):
    print(f"Pair {i+1} | Similarity: {p['question_similarity']:.0%}")
    print(f"  Original:   {p['original_question'][:70]}...")
    print(f"  Paraphrase: {p['paraphrased_question'][:70]}...")
    
    # Quick sanity check - do they share words?
    orig_words = set(p['original_question'].lower().split()[:5])
    para_words = set(p['paraphrased_question'].lower().split()[:5])
    overlap = orig_words & para_words
    print(f"  Word overlap: {overlap}")
    print()

# Check for any mismatches (similarity should be > 0.6)
low_sim = [p for p in pairs if p['question_similarity'] < 0.6]
print(f"Pairs with low similarity (<0.6): {len(low_sim)}")

# Check for obvious mismatches like "Bengal cat" vs "Bible"
suspicious = []
for p in pairs:
    orig = p['original_question'].lower()
    para = p['paraphrased_question'].lower()
    # Check if completely unrelated topics
    if ('cat' in para and 'cat' not in orig) or ('bible' in orig and 'bible' not in para):
        suspicious.append(p)

print(f"Suspicious mismatches found: {len(suspicious)}")
if suspicious:
    for s in suspicious[:3]:
        print(f"  - {s['original_question'][:50]}... vs {s['paraphrased_question'][:50]}...")