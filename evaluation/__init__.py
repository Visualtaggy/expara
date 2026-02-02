"""
ExPara Evaluation Module
"""

from .explanation_generator import (
    ExplanationGenerator,
    GenerationConfig,
    check_answer_match
)

from .pipeline import (
    EvaluationPipeline,
    EvaluationConfig,
    run_full_evaluation
)

__all__ = [
    "ExplanationGenerator",
    "GenerationConfig",
    "check_answer_match",
    "EvaluationPipeline",
    "EvaluationConfig",
    "run_full_evaluation"
]
