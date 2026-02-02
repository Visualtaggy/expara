"""
ExPara: Explanation Paraphrase Consistency Benchmark

A framework for measuring whether LLM explanations remain semantically
consistent when questions are paraphrased.

Main components:
- data: Dataset loading, paraphrase generation, data models
- metrics: ExPara-Sim consistency metrics
- evaluation: Explanation generation and evaluation pipeline
- training: Consistency-aware training methods
- utils: Helper functions

Quick Start:
    from expara.data import DatasetLoader, ParaphraseGenerator
    from expara.evaluation import ExplanationGenerator, EvaluationPipeline
    from expara.metrics import ExParaSimScore
    
    # Load data
    loader = DatasetLoader()
    questions = loader.load_gsm8k(num_samples=100)
    
    # Generate paraphrases
    generator = ParaphraseGenerator()
    paraphrases = await generator.generate_paraphrases_batch(questions)
    
    # Evaluate consistency
    scorer = ExParaSimScore()
    score, metrics = scorer.compute(explanation1, explanation2)
"""

__version__ = "0.1.0"
__author__ = "ExPara Team"

from . import data
from . import metrics
from . import evaluation
from . import training
from . import utils
