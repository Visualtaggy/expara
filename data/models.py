"""
ExPara Data Models and Schemas
Defines the core data structures used throughout the project.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class ParaphraseType(Enum):
    """Types of paraphrases in the benchmark."""
    LEXICAL = "lexical"          # Synonym substitution
    SYNTACTIC = "syntactic"      # Sentence restructuring
    VOICE = "voice"              # Active/passive transformation
    ELABORATIVE = "elaborative"  # Added context, same meaning
    MIXED = "mixed"              # Combination of types


class TaskType(Enum):
    """Types of reasoning tasks."""
    MATH = "math"
    COMMONSENSE = "commonsense"
    SCIENCE = "science"
    LOGIC = "logic"


@dataclass
class Question:
    """Represents a base question from the source dataset."""
    id: str
    text: str
    answer: str
    task_type: TaskType
    source_dataset: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "answer": self.answer,
            "task_type": self.task_type.value,
            "source_dataset": self.source_dataset,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Question":
        return cls(
            id=data["id"],
            text=data["text"],
            answer=data["answer"],
            task_type=TaskType(data["task_type"]),
            source_dataset=data["source_dataset"],
            metadata=data.get("metadata", {})
        )


@dataclass
class Paraphrase:
    """Represents a paraphrase of a question."""
    id: str
    original_question_id: str
    text: str
    paraphrase_type: ParaphraseType
    generator_model: str
    
    # Quality metrics
    nli_score: float = 0.0          # Bidirectional entailment score
    bleu_to_original: float = 0.0   # Surface similarity to original
    
    # Validation
    is_valid: bool = True
    validation_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "original_question_id": self.original_question_id,
            "text": self.text,
            "paraphrase_type": self.paraphrase_type.value,
            "generator_model": self.generator_model,
            "nli_score": self.nli_score,
            "bleu_to_original": self.bleu_to_original,
            "is_valid": self.is_valid,
            "validation_notes": self.validation_notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Paraphrase":
        return cls(
            id=data["id"],
            original_question_id=data["original_question_id"],
            text=data["text"],
            paraphrase_type=ParaphraseType(data["paraphrase_type"]),
            generator_model=data["generator_model"],
            nli_score=data.get("nli_score", 0.0),
            bleu_to_original=data.get("bleu_to_original", 0.0),
            is_valid=data.get("is_valid", True),
            validation_notes=data.get("validation_notes", "")
        )


@dataclass
class Explanation:
    """Represents a chain-of-thought explanation for a question."""
    id: str
    question_id: str  # Can be original question or paraphrase
    question_text: str
    
    # The explanation itself
    reasoning: str      # The CoT reasoning trace
    final_answer: str   # The extracted final answer
    
    # Generation metadata
    model: str
    prompt_type: str    # "zero_shot_cot", "few_shot_cot", etc.
    temperature: float
    
    # Answer correctness
    is_correct: Optional[bool] = None
    
    # For analysis
    num_reasoning_steps: int = 0
    reasoning_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer,
            "model": self.model,
            "prompt_type": self.prompt_type,
            "temperature": self.temperature,
            "is_correct": self.is_correct,
            "num_reasoning_steps": self.num_reasoning_steps,
            "reasoning_steps": self.reasoning_steps
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Explanation":
        return cls(
            id=data["id"],
            question_id=data["question_id"],
            question_text=data["question_text"],
            reasoning=data["reasoning"],
            final_answer=data["final_answer"],
            model=data["model"],
            prompt_type=data["prompt_type"],
            temperature=data["temperature"],
            is_correct=data.get("is_correct"),
            num_reasoning_steps=data.get("num_reasoning_steps", 0),
            reasoning_steps=data.get("reasoning_steps", [])
        )


@dataclass
class ExplanationPair:
    """A pair of explanations for consistency comparison."""
    id: str
    original_question: Question
    paraphrase: Paraphrase
    
    explanation_original: Explanation
    explanation_paraphrase: Explanation
    
    # Consistency metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    expara_sim_score: Optional[float] = None
    
    # Analysis flags
    same_answer: bool = False
    same_correctness: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "original_question": self.original_question.to_dict(),
            "paraphrase": self.paraphrase.to_dict(),
            "explanation_original": self.explanation_original.to_dict(),
            "explanation_paraphrase": self.explanation_paraphrase.to_dict(),
            "metrics": self.metrics,
            "expara_sim_score": self.expara_sim_score,
            "same_answer": self.same_answer,
            "same_correctness": self.same_correctness
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ExplanationPair":
        return cls(
            id=data["id"],
            original_question=Question.from_dict(data["original_question"]),
            paraphrase=Paraphrase.from_dict(data["paraphrase"]),
            explanation_original=Explanation.from_dict(data["explanation_original"]),
            explanation_paraphrase=Explanation.from_dict(data["explanation_paraphrase"]),
            metrics=data.get("metrics", {}),
            expara_sim_score=data.get("expara_sim_score"),
            same_answer=data.get("same_answer", False),
            same_correctness=data.get("same_correctness", False)
        )


@dataclass
class HumanAnnotation:
    """Human annotation for explanation consistency."""
    pair_id: str
    annotator_id: str
    
    # Main judgment (1-5 scale)
    consistency_score: int  # 1=completely different, 5=semantically equivalent
    
    # Detailed annotations
    same_key_reasoning: bool
    same_facts_used: bool
    same_logical_structure: bool
    
    # Free-form notes
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "pair_id": self.pair_id,
            "annotator_id": self.annotator_id,
            "consistency_score": self.consistency_score,
            "same_key_reasoning": self.same_key_reasoning,
            "same_facts_used": self.same_facts_used,
            "same_logical_structure": self.same_logical_structure,
            "notes": self.notes
        }


@dataclass
class EvaluationResult:
    """Results from evaluating a model on the ExPara benchmark."""
    model: str
    prompt_type: str
    temperature: float
    
    # Overall metrics
    num_pairs: int
    avg_expara_sim: float
    std_expara_sim: float
    
    # Per-metric averages
    avg_semantic_sim: float
    avg_entailment_sym: float
    avg_step_align: float
    avg_fact_overlap: float
    avg_llm_judge: float
    
    # Answer consistency
    answer_consistency_rate: float
    
    # Breakdown by task type
    by_task_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Breakdown by paraphrase type
    by_paraphrase_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "prompt_type": self.prompt_type,
            "temperature": self.temperature,
            "num_pairs": self.num_pairs,
            "avg_expara_sim": self.avg_expara_sim,
            "std_expara_sim": self.std_expara_sim,
            "avg_semantic_sim": self.avg_semantic_sim,
            "avg_entailment_sym": self.avg_entailment_sym,
            "avg_step_align": self.avg_step_align,
            "avg_fact_overlap": self.avg_fact_overlap,
            "avg_llm_judge": self.avg_llm_judge,
            "answer_consistency_rate": self.answer_consistency_rate,
            "by_task_type": self.by_task_type,
            "by_paraphrase_type": self.by_paraphrase_type
        }


# Utility functions for serialization
def save_jsonl(data: List[Any], filepath: str):
    """Save a list of dataclass objects to JSONL."""
    with open(filepath, 'w') as f:
        for item in data:
            if hasattr(item, 'to_dict'):
                f.write(json.dumps(item.to_dict()) + '\n')
            else:
                f.write(json.dumps(item) + '\n')


def load_jsonl(filepath: str, cls: type) -> List[Any]:
    """Load JSONL file into a list of dataclass objects."""
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if hasattr(cls, 'from_dict'):
                items.append(cls.from_dict(data))
            else:
                items.append(data)
    return items
