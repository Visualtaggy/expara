"""
ExPara Dataset Loader
Loads and processes source datasets (GSM8K, StrategyQA, ARC, LogiQA).
"""

import random
from typing import List, Dict, Optional, Iterator
from datasets import load_dataset
import logging

from .models import Question, TaskType

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads questions from various reasoning datasets."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_gsm8k(self, num_samples: int = 1000) -> List[Question]:
        """Load questions from GSM8K (math word problems)."""
        logger.info(f"Loading GSM8K dataset, sampling {num_samples} questions")
        
        dataset = load_dataset("gsm8k", "main", split="train")
        
        # Sample if needed
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)
        
        questions = []
        for i, item in enumerate(dataset):
            # Extract the final answer from the solution
            solution = item["answer"]
            # GSM8K format: solution ends with "#### <answer>"
            if "####" in solution:
                answer = solution.split("####")[-1].strip()
            else:
                answer = solution.strip()
            
            q = Question(
                id=f"gsm8k_{i}",
                text=item["question"],
                answer=answer,
                task_type=TaskType.MATH,
                source_dataset="gsm8k",
                metadata={"full_solution": solution}
            )
            questions.append(q)
        
        logger.info(f"Loaded {len(questions)} questions from GSM8K")
        return questions
    
    def load_strategyqa(self, num_samples: int = 1000) -> List[Question]:
        """Load questions from StrategyQA (commonsense reasoning)."""
        logger.info(f"Loading StrategyQA dataset, sampling {num_samples} questions")
        
        try:
            dataset = load_dataset("wics/strategy-qa", split="train")
        except Exception:
            # Fallback to alternative source
            dataset = load_dataset("tasksource/strategy-qa", split="train")
        
        # Sample if needed
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)
        
        questions = []
        for i, item in enumerate(dataset):
            answer = "Yes" if item.get("answer", item.get("label", False)) else "No"
            
            q = Question(
                id=f"strategyqa_{i}",
                text=item["question"],
                answer=answer,
                task_type=TaskType.COMMONSENSE,
                source_dataset="strategyqa",
                metadata={
                    "facts": item.get("facts", []),
                    "decomposition": item.get("decomposition", [])
                }
            )
            questions.append(q)
        
        logger.info(f"Loaded {len(questions)} questions from StrategyQA")
        return questions
    
    def load_arc_challenge(self, num_samples: int = 1000) -> List[Question]:
        """Load questions from ARC-Challenge (science questions)."""
        logger.info(f"Loading ARC-Challenge dataset, sampling {num_samples} questions")
        
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        
        # Sample if needed
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)
        
        questions = []
        for i, item in enumerate(dataset):
            # Format question with choices
            choices = item["choices"]
            choice_text = "\n".join([
                f"{label}. {text}" 
                for label, text in zip(choices["label"], choices["text"])
            ])
            full_question = f"{item['question']}\n\n{choice_text}"
            
            # Get correct answer
            answer_key = item["answerKey"]
            answer_idx = choices["label"].index(answer_key)
            answer = choices["text"][answer_idx]
            
            q = Question(
                id=f"arc_{i}",
                text=full_question,
                answer=f"{answer_key}. {answer}",
                task_type=TaskType.SCIENCE,
                source_dataset="arc_challenge",
                metadata={
                    "answer_key": answer_key,
                    "choices": dict(zip(choices["label"], choices["text"]))
                }
            )
            questions.append(q)
        
        logger.info(f"Loaded {len(questions)} questions from ARC-Challenge")
        return questions
    
    def load_logiqa(self, num_samples: int = 1000) -> List[Question]:
        """Load questions from LogiQA (logical reasoning)."""
        logger.info(f"Loading LogiQA dataset, sampling {num_samples} questions")
        
        dataset = None
        
        # Try multiple sources
        sources = [
            ("Nan-Do/logiqa", None),
            ("tasksource/logiqa", None),
        ]
        
        for source, config in sources:
            try:
                if config:
                    dataset = load_dataset(source, config, split="train", trust_remote_code=True)
                else:
                    dataset = load_dataset(source, split="train", trust_remote_code=True)
                logger.info(f"Successfully loaded LogiQA from {source}")
                break
            except Exception as e:
                logger.debug(f"Failed to load from {source}: {e}")
                continue
        
        if dataset is None:
            # Final fallback - create synthetic logic questions
            logger.warning("LogiQA not available, using synthetic logic questions")
            return self._create_synthetic_logic_questions(num_samples)
        
        # Sample if needed
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)
        
        questions = []
        for i, item in enumerate(dataset):
            try:
                # Handle different dataset formats
                context = item.get("context", item.get("passage", item.get("text", "")))
                question_text = item.get("question", item.get("query", ""))
                options = item.get("options", item.get("choices", item.get("answers", [])))
                
                # Handle options format
                if isinstance(options, dict):
                    options = list(options.values())
                elif not isinstance(options, list):
                    options = []
                
                if options:
                    options_text = "\n".join([
                        f"{chr(65+j)}. {opt}" for j, opt in enumerate(options)
                    ])
                else:
                    options_text = ""
                
                full_question = f"{context}\n\nQuestion: {question_text}\n\n{options_text}".strip()
                
                # Get answer
                label = item.get("label", item.get("answer", 0))
                if isinstance(label, int) and options and label < len(options):
                    answer = f"{chr(65+label)}. {options[label]}"
                elif isinstance(label, str) and len(label) == 1 and label.upper() in "ABCD":
                    idx = ord(label.upper()) - ord('A')
                    if options and idx < len(options):
                        answer = f"{label.upper()}. {options[idx]}"
                    else:
                        answer = label
                else:
                    answer = str(label)
                
                q = Question(
                    id=f"logiqa_{i}",
                    text=full_question,
                    answer=answer,
                    task_type=TaskType.LOGIC,
                    source_dataset="logiqa",
                    metadata={"label": label}
                )
                questions.append(q)
            except Exception as e:
                logger.debug(f"Skipping item {i} due to error: {e}")
                continue
        
        logger.info(f"Loaded {len(questions)} questions from LogiQA")
        return questions
    
    def _create_synthetic_logic_questions(self, num_samples: int) -> List[Question]:
        """Create synthetic logic questions as fallback."""
        templates = [
            {
                "context": "All mammals are warm-blooded. All dogs are mammals.",
                "question": "Based on the above, which of the following must be true?",
                "options": ["All dogs are warm-blooded", "Some warm-blooded animals are not mammals", 
                           "No dogs are warm-blooded", "Some mammals are not dogs"],
                "answer": 0
            },
            {
                "context": "If it rains, the ground gets wet. The ground is wet.",
                "question": "What can we conclude?",
                "options": ["It definitely rained", "It might have rained", 
                           "It did not rain", "The ground is dry"],
                "answer": 1
            },
            {
                "context": "No birds are mammals. Some pets are birds.",
                "question": "Which conclusion follows?",
                "options": ["All pets are mammals", "No pets are mammals",
                           "Some pets are not mammals", "All birds are pets"],
                "answer": 2
            },
            {
                "context": "Either John went to the store or Mary stayed home. John did not go to the store.",
                "question": "What must be true?",
                "options": ["Mary went to the store", "Mary stayed home",
                           "John stayed home", "Both stayed home"],
                "answer": 1
            },
            {
                "context": "All squares are rectangles. All rectangles have four sides.",
                "question": "What can we conclude about squares?",
                "options": ["Squares have three sides", "Squares have four sides",
                           "Squares are not rectangles", "Rectangles are squares"],
                "answer": 1
            },
        ]
        
        questions = []
        for i in range(min(num_samples, len(templates) * 10)):
            template = templates[i % len(templates)]
            options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(template["options"])])
            
            q = Question(
                id=f"logiqa_synthetic_{i}",
                text=f"{template['context']}\n\nQuestion: {template['question']}\n\n{options_text}",
                answer=f"{chr(65+template['answer'])}. {template['options'][template['answer']]}",
                task_type=TaskType.LOGIC,
                source_dataset="logiqa_synthetic",
                metadata={"label": template["answer"]}
            )
            questions.append(q)
        
        logger.info(f"Created {len(questions)} synthetic logic questions")
        return questions
    
    def load_all(self, samples_per_dataset: int = 1000) -> Dict[str, List[Question]]:
        """Load all datasets."""
        return {
            "gsm8k": self.load_gsm8k(samples_per_dataset),
            "strategyqa": self.load_strategyqa(samples_per_dataset),
            "arc_challenge": self.load_arc_challenge(samples_per_dataset),
            "logiqa": self.load_logiqa(samples_per_dataset)
        }
    
    def load_combined(self, samples_per_dataset: int = 1000) -> List[Question]:
        """Load all datasets and combine into a single list."""
        all_data = self.load_all(samples_per_dataset)
        combined = []
        for questions in all_data.values():
            combined.extend(questions)
        random.shuffle(combined)
        return combined


def iterate_questions(questions: List[Question], batch_size: int = 32) -> Iterator[List[Question]]:
    """Iterate over questions in batches."""
    for i in range(0, len(questions), batch_size):
        yield questions[i:i + batch_size]


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    loader = DatasetLoader()
    
    # Load small samples for testing
    gsm8k = loader.load_gsm8k(num_samples=5)
    print(f"\nGSM8K sample:")
    print(f"  Q: {gsm8k[0].text[:100]}...")
    print(f"  A: {gsm8k[0].answer}")
    
    strategyqa = loader.load_strategyqa(num_samples=5)
    print(f"\nStrategyQA sample:")
    print(f"  Q: {strategyqa[0].text}")
    print(f"  A: {strategyqa[0].answer}")
