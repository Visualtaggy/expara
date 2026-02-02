"""
ExPara Utilities
Helper functions and classes.
"""

import os
import json
import random
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_id(*args) -> str:
    """Generate a unique ID from input arguments."""
    content = "_".join(str(a) for a in args)
    return hashlib.md5(content.encode()).hexdigest()[:12]


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: str):
    """Save configuration to YAML file."""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class ResultsTracker:
    """Track and aggregate results across experiments."""
    
    def __init__(self, output_dir: str):
        self.output_dir = ensure_dir(output_dir)
        self.results = []
    
    def add_result(self, result: Dict, name: str = None):
        """Add a result to the tracker."""
        if name:
            result["name"] = name
        self.results.append(result)
    
    def save(self, filename: str = "all_results.json"):
        """Save all results to file."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Saved results to {path}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all results."""
        if not self.results:
            return {}
        
        import numpy as np
        
        # Extract numeric metrics
        metrics = ["avg_expara_sim", "avg_semantic_sim", "avg_entailment_sym",
                   "avg_step_align", "avg_fact_overlap", "avg_llm_judge",
                   "answer_consistency_rate"]
        
        summary = {}
        for metric in metrics:
            values = [r.get(metric) for r in self.results if r.get(metric) is not None]
            if values:
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return summary


class ProgressLogger:
    """Simple progress logger for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        pct = (self.current / self.total) * 100
        logger.info(f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%)")
    
    def done(self):
        """Mark as complete."""
        logger.info(f"{self.desc}: Complete!")


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def extract_answer_from_text(text: str, task_type: str = None) -> str:
    """Extract the final answer from explanation text."""
    import re
    
    # Common patterns
    patterns = [
        r"(?:Therefore|Thus|So|Hence)[,:]?\s*(?:the answer is\s*)?(.+?)(?:\.|$)",
        r"(?:The answer is|Answer:)\s*(.+?)(?:\.|$)",
        r"####\s*(.+?)$",
        r"=\s*(\d+(?:\.\d+)?)\s*$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Fallback: last line
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    import re
    
    # Lowercase and strip
    answer = answer.lower().strip()
    
    # Remove common prefixes
    for prefix in ["the answer is", "answer:", "therefore,"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove punctuation at end
    answer = re.sub(r'[.,!?]+$', '', answer)
    
    return answer


def compare_answers(answer1: str, answer2: str, strict: bool = False) -> bool:
    """Compare two answers for equivalence."""
    import re
    
    a1 = normalize_answer(answer1)
    a2 = normalize_answer(answer2)
    
    if strict:
        return a1 == a2
    
    # Flexible matching
    if a1 == a2:
        return True
    if a1 in a2 or a2 in a1:
        return True
    
    # Numeric comparison
    nums1 = re.findall(r'[\d,]+\.?\d*', a1)
    nums2 = re.findall(r'[\d,]+\.?\d*', a2)
    
    if nums1 and nums2:
        try:
            v1 = float(nums1[-1].replace(',', ''))
            v2 = float(nums2[-1].replace(',', ''))
            if abs(v1 - v2) < 0.01:
                return True
        except ValueError:
            pass
    
    return False


def batch_iterator(items: List[Any], batch_size: int):
    """Iterate over items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
