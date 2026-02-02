"""
ExPara Explanation Generator (Open-Source Version)
Generates chain-of-thought explanations using local HuggingFace models.
NO API KEYS REQUIRED!
"""

import hashlib
import re
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from data.models import Question, Paraphrase, Explanation

logger = logging.getLogger(__name__)


# CoT Prompts
ZERO_SHOT_COT_PROMPT = """Answer the following question. Think step by step and show your reasoning.

Question: {question}

Let's think step by step:"""

FEW_SHOT_COT_PROMPT = """Answer questions by thinking step by step.

{examples}

Question: {question}

Let's think step by step:"""

# Few-shot examples by task type
FEW_SHOT_EXAMPLES = {
    "math": """Question: If John has 5 apples and gives 2 to Mary, how many apples does John have?

Let's think step by step:
1. John starts with 5 apples
2. John gives 2 apples to Mary
3. 5 - 2 = 3 apples remaining

Therefore, John has 3 apples.

---

Question: A store sells pencils for $0.50 each. How much do 6 pencils cost?

Let's think step by step:
1. Each pencil costs $0.50
2. We need to find the cost of 6 pencils
3. 6 × $0.50 = $3.00

Therefore, 6 pencils cost $3.00.""",

    "commonsense": """Question: Can a fish climb a tree?

Let's think step by step:
1. Fish are aquatic animals that live in water
2. Fish have fins, not limbs suitable for climbing
3. Trees are land-based structures
4. Fish cannot survive long outside water and lack climbing ability

Therefore, no, a fish cannot climb a tree.

---

Question: Is it possible to see the sun at night?

Let's think step by step:
1. "Night" is defined as the period when your location faces away from the sun
2. During night, the sun is below the horizon
3. The sun itself cannot be seen directly at night from a normal location
4. However, you could see sunlight reflected off the moon

Therefore, no, you cannot see the sun directly at night.""",

    "science": """Question: Why do objects fall to the ground when dropped?

Let's think step by step:
1. Earth has mass, which creates a gravitational field
2. This gravitational force attracts all objects with mass toward Earth's center
3. When an object is dropped, there's no support force to counteract gravity
4. The object accelerates toward the ground at approximately 9.8 m/s²

Therefore, objects fall due to Earth's gravitational pull.

---

Question: Why does ice float on water?

Let's think step by step:
1. Ice is the solid form of water
2. When water freezes, its molecules form a crystalline structure
3. This structure is less dense than liquid water
4. Objects less dense than water float

Therefore, ice floats because it is less dense than liquid water.""",

    "logic": """Question: If all cats are animals, and some animals are pets, can we conclude that some cats are pets?

Let's think step by step:
1. Premise 1: All cats are animals (Cats ⊆ Animals)
2. Premise 2: Some animals are pets (Animals ∩ Pets ≠ ∅)
3. From premise 2, we know there exist some animals that are pets
4. However, we don't know if those animals that are pets include cats
5. The intersection could be with non-cat animals

Therefore, no, we cannot logically conclude that some cats are pets.

---

Question: If it's raining, then the ground is wet. The ground is wet. Is it raining?

Let's think step by step:
1. We have: If rain → wet ground
2. We observe: wet ground
3. This is the fallacy of affirming the consequent
4. The ground could be wet for other reasons (sprinklers, spill, etc.)
5. We cannot conclude it's raining just because the ground is wet

Therefore, we cannot conclude it is raining."""
}


@dataclass
class GenerationConfig:
    """Configuration for explanation generation."""
    prompt_type: str = "zero_shot_cot"  # or "few_shot_cot"
    temperature: float = 0.0
    max_new_tokens: int = 512
    top_p: float = 0.9
    do_sample: bool = False  # False for temperature=0


class HuggingFaceExplanationGenerator:
    """
    Generate CoT explanations using open-source HuggingFace models.
    No API keys needed!
    """
    
    # Recommended models for CoT reasoning
    RECOMMENDED_MODELS = [
        # Best reasoning models
        "Qwen/Qwen2.5-7B-Instruct",              # Excellent reasoning
        "Qwen/Qwen2.5-3B-Instruct",              # Good balance
        "meta-llama/Llama-3.1-8B-Instruct",      # Strong performer
        "mistralai/Mistral-7B-Instruct-v0.3",    # Good reasoning
        "microsoft/Phi-3-mini-4k-instruct",      # Fast, decent
        "google/gemma-2-9b-it",                  # Good quality
        "google/gemma-2-2b-it",                  # Smallest option
        
        # Specialized reasoning models (if available)
        "deepseek-ai/deepseek-math-7b-instruct", # Math specialized
        "TIGER-Lab/MAmmoTH2-8B",                 # Math reasoning
    ]
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto"
    ):
        """
        Initialize the explanation generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or None for auto)
            load_in_4bit: Use 4-bit quantization (saves ~75% memory)
            load_in_8bit: Use 8-bit quantization (saves ~50% memory)
            torch_dtype: Data type ("auto", "float16", "bfloat16")
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        # Determine dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare model loading kwargs
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True
            )
            load_kwargs["device_map"] = "auto"
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = dtype
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        # Move to device if not using device_map
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully! Memory: {self._get_memory_usage()}")
    
    def _get_memory_usage(self) -> str:
        """Get current GPU memory usage."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            return f"{allocated:.2f} GB"
        return "N/A (CPU)"
    
    def _build_prompt(
        self,
        question_text: str,
        task_type: str,
        config: GenerationConfig
    ) -> str:
        """Build the prompt for explanation generation."""
        if config.prompt_type == "zero_shot_cot":
            return ZERO_SHOT_COT_PROMPT.format(question=question_text)
        elif config.prompt_type == "few_shot_cot":
            examples = FEW_SHOT_EXAMPLES.get(task_type, FEW_SHOT_EXAMPLES["commonsense"])
            return FEW_SHOT_COT_PROMPT.format(examples=examples, question=question_text)
        else:
            raise ValueError(f"Unknown prompt type: {config.prompt_type}")
    
    def _format_chat(self, prompt: str) -> str:
        """Format prompt as chat message if supported."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> str:
        """Generate explanation from prompt."""
        formatted = self._format_chat(prompt)
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
        else:
            gen_kwargs["do_sample"] = False
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_explanation(
        self,
        question_text: str,
        question_id: str,
        task_type: str,
        config: GenerationConfig
    ) -> Explanation:
        """Generate a single explanation."""
        prompt = self._build_prompt(question_text, task_type, config)
        reasoning = self.generate(prompt, config)
        
        # Extract final answer and steps
        final_answer, reasoning_steps = self._parse_explanation(reasoning)
        
        # Generate ID
        exp_id = hashlib.md5(
            f"{question_id}_{self.model_name}_{config.prompt_type}_{config.temperature}".encode()
        ).hexdigest()[:12]
        
        return Explanation(
            id=exp_id,
            question_id=question_id,
            question_text=question_text,
            reasoning=reasoning,
            final_answer=final_answer,
            model=self.model_name,
            prompt_type=config.prompt_type,
            temperature=config.temperature,
            num_reasoning_steps=len(reasoning_steps),
            reasoning_steps=reasoning_steps
        )
    
    def _parse_explanation(self, reasoning: str) -> Tuple[str, List[str]]:
        """Parse explanation to extract final answer and steps."""
        # Extract final answer
        answer_patterns = [
            r"(?:Therefore|Thus|So|Hence)[,:]?\s*(?:the answer is\s*)?(.+?)(?:\.|$)",
            r"(?:Final answer|Answer)[,:]?\s*(.+?)(?:\.|$)",
            r"####\s*(.+?)$",
            r"The answer is[:\s]+(.+?)(?:\.|$)",
        ]
        
        final_answer = ""
        for pattern in answer_patterns:
            match = re.search(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
            if match:
                final_answer = match.group(1).strip()
                break
        
        if not final_answer:
            sentences = reasoning.split(".")
            if sentences:
                final_answer = sentences[-1].strip() or (sentences[-2].strip() if len(sentences) > 1 else "")
        
        # Extract reasoning steps
        numbered_pattern = r"(?:^|\n)\s*(\d+[\.\)]\s*.+?)(?=\n\s*\d+[\.\)]|\n\n|$)"
        matches = re.findall(numbered_pattern, reasoning, re.MULTILINE | re.DOTALL)
        
        if matches:
            steps = [m.strip() for m in matches]
        else:
            lines = reasoning.split("\n")
            steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        
        return final_answer, steps
    
    def generate_explanations_batch(
        self,
        questions: List[Union[Question, Paraphrase]],
        original_questions: Optional[Dict[str, Question]] = None,
        config: Optional[GenerationConfig] = None,
        show_progress: bool = True
    ) -> List[Explanation]:
        """Generate explanations for a batch of questions."""
        if config is None:
            config = GenerationConfig()
        
        explanations = []
        iterator = tqdm(questions, desc="Generating explanations") if show_progress else questions
        
        for q in iterator:
            try:
                # Determine task type
                if isinstance(q, Question):
                    task_type = q.task_type.value
                    q_id = q.id
                    q_text = q.text
                else:
                    # Paraphrase
                    if original_questions and q.original_question_id in original_questions:
                        task_type = original_questions[q.original_question_id].task_type.value
                    else:
                        task_type = "commonsense"
                    q_id = q.id
                    q_text = q.text
                
                explanation = self.generate_explanation(q_text, q_id, task_type, config)
                explanations.append(explanation)
                
            except Exception as e:
                logger.error(f"Error generating explanation for {q.id}: {e}")
        
        return explanations


class ExplanationGenerator:
    """
    High-level explanation generator with multiple backend support.
    Defaults to open-source models - NO API keys needed!
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = None,
        load_in_4bit: bool = False,
        use_mock: bool = False
    ):
        """
        Initialize the explanation generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            use_mock: Use mock generator for testing
        """
        self.use_mock = use_mock
        self.model_name = model_name
        
        if use_mock:
            logger.info("Using mock explanation generator")
            self.generator = None
        else:
            self.generator = HuggingFaceExplanationGenerator(
                model_name=model_name,
                device=device,
                load_in_4bit=load_in_4bit
            )
    
    def _mock_explanation(self, question: str, task_type: str) -> str:
        """Generate mock explanation for testing."""
        return f"""Let me think through this step by step:

1. First, I'll analyze the question: "{question[:50]}..."
2. This appears to be a {task_type} question
3. Based on my reasoning, I need to consider the key facts
4. After careful analysis of the problem

Therefore, the answer is: [Mock Answer]"""
    
    def generate_explanation(
        self,
        question_text: str,
        question_id: str,
        task_type: str,
        config: Optional[GenerationConfig] = None
    ) -> Explanation:
        """Generate a single explanation."""
        if config is None:
            config = GenerationConfig()
        
        if self.use_mock:
            reasoning = self._mock_explanation(question_text, task_type)
            return Explanation(
                id=f"mock_{question_id}",
                question_id=question_id,
                question_text=question_text,
                reasoning=reasoning,
                final_answer="[Mock Answer]",
                model="mock",
                prompt_type=config.prompt_type,
                temperature=config.temperature,
                num_reasoning_steps=4,
                reasoning_steps=reasoning.split("\n")
            )
        
        return self.generator.generate_explanation(
            question_text, question_id, task_type, config
        )
    
    def generate_explanations_batch(
        self,
        questions: List[Union[Question, Paraphrase]],
        original_questions: Optional[Dict[str, Question]] = None,
        config: Optional[GenerationConfig] = None,
        show_progress: bool = True
    ) -> List[Explanation]:
        """Generate explanations for a batch."""
        if config is None:
            config = GenerationConfig()
        
        if self.use_mock:
            explanations = []
            iterator = tqdm(questions, desc="Generating (mock)") if show_progress else questions
            
            for q in iterator:
                if isinstance(q, Question):
                    task_type = q.task_type.value
                else:
                    task_type = "commonsense"
                
                exp = self.generate_explanation(q.text, q.id, task_type, config)
                explanations.append(exp)
            
            return explanations
        
        return self.generator.generate_explanations_batch(
            questions, original_questions, config, show_progress
        )


def check_answer_match(
    explanation: Explanation,
    ground_truth: str,
    task_type: str
) -> bool:
    """Check if explanation's answer matches ground truth."""
    pred = explanation.final_answer.lower().strip()
    truth = ground_truth.lower().strip()
    
    if pred == truth or truth in pred or pred in truth:
        return True
    
    # Numeric
    if task_type == "math":
        pred_nums = re.findall(r'[\d,]+\.?\d*', pred)
        truth_nums = re.findall(r'[\d,]+\.?\d*', truth)
        if pred_nums and truth_nums:
            try:
                pred_val = float(pred_nums[-1].replace(",", ""))
                truth_val = float(truth_nums[-1].replace(",", ""))
                return abs(pred_val - truth_val) < 0.01
            except ValueError:
                pass
    
    # Yes/No
    if task_type == "commonsense":
        pred_yn = "yes" if "yes" in pred else ("no" if "no" in pred else None)
        truth_yn = "yes" if "yes" in truth else ("no" if "no" in truth else None)
        if pred_yn and truth_yn:
            return pred_yn == truth_yn
    
    # Multiple choice
    mc_pattern = r"^([A-D])[\.\)]"
    pred_mc = re.match(mc_pattern, pred.upper())
    truth_mc = re.match(mc_pattern, truth.upper())
    if pred_mc and truth_mc:
        return pred_mc.group(1) == truth_mc.group(1)
    
    return False


# Convenience function
def create_explanation_generator(
    model: str = "auto",
    device: str = None,
    quantize: bool = False
) -> ExplanationGenerator:
    """
    Create an explanation generator with smart defaults.
    
    Args:
        model: Model name or "auto"
        device: Device to use
        quantize: Use 4-bit quantization
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model == "auto":
        if device == "cuda":
            try:
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                if mem >= 24:
                    model = "Qwen/Qwen2.5-14B-Instruct"
                elif mem >= 16:
                    model = "Qwen/Qwen2.5-7B-Instruct"
                elif mem >= 8:
                    model = "Qwen/Qwen2.5-3B-Instruct"
                    quantize = True
                else:
                    model = "google/gemma-2-2b-it"
                    quantize = True
            except:
                model = "Qwen/Qwen2.5-3B-Instruct"
        else:
            model = "google/gemma-2-2b-it"
    
    logger.info(f"Selected model: {model}, quantize: {quantize}")
    
    return ExplanationGenerator(
        model_name=model,
        device=device,
        load_in_4bit=quantize
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("ExPara Explanation Generator (Open-Source)")
    print("=" * 50)
    print("\nRecommended models:")
    for m in HuggingFaceExplanationGenerator.RECOMMENDED_MODELS[:5]:
        print(f"  - {m}")
    
    print("\nUsage:")
    print("  # Auto-select best model for your GPU")
    print("  generator = create_explanation_generator(model='auto')")
    print("")
    print("  # Or specify model")
    print("  generator = ExplanationGenerator(model_name='Qwen/Qwen2.5-7B-Instruct')")
    print("")
    print("  # For limited memory (4-bit quantization)")
    print("  generator = ExplanationGenerator(model_name='Qwen/Qwen2.5-7B-Instruct', load_in_4bit=True)")
    print("")
    print("  # For testing without GPU")
    print("  generator = ExplanationGenerator(use_mock=True)")
