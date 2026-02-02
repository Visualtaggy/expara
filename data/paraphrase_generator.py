"""
ExPara Paraphrase Generator (Open-Source Version)
Generates diverse paraphrases using local HuggingFace models - NO API keys needed.
"""

import asyncio
import random
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .models import Question, Paraphrase, ParaphraseType

logger = logging.getLogger(__name__)


# Prompts for different paraphrase types
PARAPHRASE_PROMPTS = {
    ParaphraseType.LEXICAL: """Paraphrase the following question by replacing words with synonyms. 
Keep the exact same meaning but use different vocabulary.
Do NOT change the structure of the sentence.
Do NOT add or remove any information.

Original question: {question}

Paraphrased question (using synonyms only):""",

    ParaphraseType.SYNTACTIC: """Paraphrase the following question by restructuring the sentence.
Keep the exact same meaning but change the grammatical structure.
You may reorder clauses, change from question to statement form (that asks), etc.
Do NOT change the vocabulary significantly.
Do NOT add or remove any information.

Original question: {question}

Paraphrased question (restructured):""",

    ParaphraseType.VOICE: """Paraphrase the following question by changing the voice (active to passive or vice versa) where applicable.
Keep the exact same meaning.
Do NOT add or remove any information.

Original question: {question}

Paraphrased question (voice changed):""",

    ParaphraseType.ELABORATIVE: """Paraphrase the following question by slightly rephrasing it in a more natural, conversational way.
Keep the EXACT same meaning - the answer should be identical.
You may add minor clarifying phrases but do NOT change what is being asked.
Do NOT add new constraints or information that would change the answer.

Original question: {question}

Paraphrased question (natural rephrasing):""",

    ParaphraseType.MIXED: """Paraphrase the following question to ask the same thing in a completely different way.
Requirements:
1. The meaning must be EXACTLY the same - the correct answer must be identical
2. Use different words and sentence structure
3. Do NOT add or remove any information that would change the answer
4. Make it sound natural

Original question: {question}

Paraphrased question:"""
}


class HuggingFaceGenerator:
    """
    Open-source LLM generator using HuggingFace transformers.
    No API keys required!
    """
    
    # Recommended open-source models (in order of quality/size tradeoff)
    RECOMMENDED_MODELS = [
        "microsoft/Phi-3-mini-4k-instruct",      # 3.8B - fast, good quality
        "Qwen/Qwen2.5-3B-Instruct",              # 3B - excellent for its size
        "Qwen/Qwen2.5-7B-Instruct",              # 7B - better quality
        "meta-llama/Llama-3.2-3B-Instruct",      # 3B - good balance
        "mistralai/Mistral-7B-Instruct-v0.3",    # 7B - strong performer
        "google/gemma-2-2b-it",                  # 2B - smallest, fastest
    ]
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = None,
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        """
        Initialize the generator with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or None for auto)
            torch_dtype: Data type ("auto", "float16", "bfloat16", "float32")
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        # Handle dtype
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
        
        # Load model with optional quantization
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        if not load_in_4bit and not load_in_8bit and self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text from prompt."""
        
        # Format as chat if model supports it
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted = prompt
        
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()


class T5ParaphraseGenerator:
    """
    Specialized paraphrase generator using T5-based models.
    These models are specifically trained for paraphrasing.
    """
    
    PARAPHRASE_MODELS = [
        "Vamsi/T5_Paraphrase_Paws",           # T5 fine-tuned on PAWS
        "humarin/chatgpt_paraphraser_on_T5_base",  # T5 paraphraser
        "tuner007/pegasus_paraphrase",        # Pegasus paraphraser
    ]
    
    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading T5 paraphrase model: {model_name}")
        
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        logger.info("T5 paraphrase model loaded!")
    
    def generate(self, text: str, num_return_sequences: int = 1) -> List[str]:
        """Generate paraphrases."""
        # T5 paraphrase models expect "paraphrase: " prefix
        if "T5" in self.pipe.model.config._name_or_path:
            input_text = f"paraphrase: {text} </s>"
        else:
            input_text = text
        
        outputs = self.pipe(
            input_text,
            max_length=256,
            num_return_sequences=num_return_sequences,
            num_beams=5,
            temperature=0.7
        )
        
        return [out["generated_text"] for out in outputs]


class ParaphraseGenerator:
    """
    Main paraphrase generator supporting multiple backends.
    Defaults to open-source models - NO API keys needed!
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        use_t5: bool = False,
        t5_model: str = "Vamsi/T5_Paraphrase_Paws",
        device: str = None,
        load_in_4bit: bool = False,
        use_mock: bool = False
    ):
        """
        Initialize paraphrase generator.
        
        Args:
            model_name: HuggingFace model for LLM-based paraphrasing
            use_t5: Use T5-based paraphrase model instead
            t5_model: T5 model name if use_t5=True
            device: Device ("cuda", "cpu", or None for auto)
            load_in_4bit: Use 4-bit quantization to save memory
            use_mock: Use mock generator for testing
        """
        self.use_mock = use_mock
        self.use_t5 = use_t5
        
        if use_mock:
            logger.info("Using mock paraphrase generator")
            self.generator = None
        elif use_t5:
            self.generator = T5ParaphraseGenerator(t5_model, device)
        else:
            self.generator = HuggingFaceGenerator(
                model_name=model_name,
                device=device,
                load_in_4bit=load_in_4bit
            )
    
    def _generate_id(self, question_id: str, paraphrase_type: str, index: int) -> str:
        """Generate a unique ID for a paraphrase."""
        content = f"{question_id}_{paraphrase_type}_{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _mock_paraphrase(self, text: str) -> str:
        """Generate a simple mock paraphrase for testing."""
        words = text.split()
        if len(words) > 4:
            # Swap some words
            i, j = random.sample(range(min(len(words), 6)), 2)
            words[i], words[j] = words[j], words[i]
        
        # Add some variation
        variations = [
            lambda t: t.replace("What", "Tell me what").replace("?", ""),
            lambda t: t.replace("How", "In what way"),
            lambda t: "Can you explain " + t.lower().rstrip("?") + "?",
            lambda t: t,
        ]
        
        result = random.choice(variations)(" ".join(words))
        return result
    
    def generate_paraphrase(
        self,
        question: Question,
        paraphrase_type: ParaphraseType,
        index: int = 0
    ) -> Paraphrase:
        """Generate a single paraphrase."""
        
        if self.use_mock:
            paraphrased_text = self._mock_paraphrase(question.text)
        elif self.use_t5:
            results = self.generator.generate(question.text, num_return_sequences=1)
            paraphrased_text = results[0] if results else question.text
        else:
            prompt = PARAPHRASE_PROMPTS[paraphrase_type].format(question=question.text)
            paraphrased_text = self.generator.generate(prompt, temperature=0.7)
            
            # Clean up output
            for prefix in ["Paraphrased question:", "Paraphrase:", "Question:", "Answer:"]:
                if paraphrased_text.lower().startswith(prefix.lower()):
                    paraphrased_text = paraphrased_text[len(prefix):].strip()
            
            # Take first line/sentence if multiple
            paraphrased_text = paraphrased_text.split('\n')[0].strip()
        
        return Paraphrase(
            id=self._generate_id(question.id, paraphrase_type.value, index),
            original_question_id=question.id,
            text=paraphrased_text,
            paraphrase_type=paraphrase_type,
            generator_model=self.generator.model_name if self.generator and hasattr(self.generator, 'model_name') else "mock",
            is_valid=bool(paraphrased_text)
        )
    
    def generate_paraphrases_for_question(
        self,
        question: Question,
        num_paraphrases: int = 4,
        paraphrase_types: Optional[List[ParaphraseType]] = None
    ) -> List[Paraphrase]:
        """Generate multiple paraphrases for a question."""
        if paraphrase_types is None:
            paraphrase_types = [
                ParaphraseType.LEXICAL,
                ParaphraseType.SYNTACTIC,
                ParaphraseType.ELABORATIVE,
                ParaphraseType.MIXED
            ]
        
        paraphrases = []
        for i in range(num_paraphrases):
            p_type = paraphrase_types[i % len(paraphrase_types)]
            
            try:
                paraphrase = self.generate_paraphrase(question, p_type, i)
                if paraphrase.is_valid and paraphrase.text:
                    paraphrases.append(paraphrase)
            except Exception as e:
                logger.error(f"Error generating paraphrase {i} for {question.id}: {e}")
        
        return paraphrases
    
    def generate_paraphrases_batch(
        self,
        questions: List[Question],
        num_paraphrases_per_question: int = 4,
        show_progress: bool = True
    ) -> Dict[str, List[Paraphrase]]:
        """Generate paraphrases for a batch of questions."""
        from tqdm import tqdm
        
        results = {}
        iterator = tqdm(questions, desc="Generating paraphrases") if show_progress else questions
        
        for question in iterator:
            paraphrases = self.generate_paraphrases_for_question(
                question, num_paraphrases_per_question
            )
            results[question.id] = paraphrases
        
        return results


class ParaphraseValidator:
    """Validates paraphrases for semantic equivalence."""
    
    def __init__(
        self,
        nli_model: str = "microsoft/deberta-v3-base-mnli",  # Smaller, faster
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_model_name = nli_model
        self._pipe = None
    
    def _load_model(self):
        """Lazy load NLI model."""
        if self._pipe is None:
            logger.info(f"Loading NLI model: {self.nli_model_name}")
            self._pipe = pipeline(
                "text-classification",
                model=self.nli_model_name,
                device=0 if self.device == "cuda" else -1
            )
    
    def compute_nli_score(self, text1: str, text2: str) -> float:
        """Compute bidirectional entailment score."""
        self._load_model()
        
        # Forward direction
        result1 = self._pipe(f"{text1} [SEP] {text2}")
        # Backward direction  
        result2 = self._pipe(f"{text2} [SEP] {text1}")
        
        # Get entailment scores
        def get_entailment(result):
            for r in result if isinstance(result, list) else [result]:
                if r['label'].lower() == 'entailment':
                    return r['score']
            return 0.0
        
        score1 = get_entailment(result1)
        score2 = get_entailment(result2)
        
        return (score1 + score2) / 2
    
    def validate_paraphrase(
        self,
        original: Question,
        paraphrase: Paraphrase,
        nli_threshold: float = 0.5
    ) -> Paraphrase:
        """Validate a paraphrase."""
        try:
            nli_score = self.compute_nli_score(original.text, paraphrase.text)
            paraphrase.nli_score = nli_score
            paraphrase.is_valid = nli_score >= nli_threshold
            
            if not paraphrase.is_valid:
                paraphrase.validation_notes = f"NLI score {nli_score:.3f} below threshold"
        except Exception as e:
            logger.error(f"Validation error: {e}")
            paraphrase.validation_notes = str(e)
        
        return paraphrase


# Convenience function for quick testing
def create_generator(
    model: str = "auto",
    device: str = None,
    quantize: bool = False
) -> ParaphraseGenerator:
    """
    Create a paraphrase generator with sensible defaults.
    
    Args:
        model: Model name or "auto" to pick based on available resources
        device: Device to use
        quantize: Use 4-bit quantization (saves memory)
    
    Returns:
        ParaphraseGenerator instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model == "auto":
        if device == "cuda":
            # Check available memory
            try:
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                if mem >= 16:
                    model = "Qwen/Qwen2.5-7B-Instruct"
                elif mem >= 8:
                    model = "Qwen/Qwen2.5-3B-Instruct"
                else:
                    model = "google/gemma-2-2b-it"
            except:
                model = "Qwen/Qwen2.5-3B-Instruct"
        else:
            # CPU - use smaller model
            model = "google/gemma-2-2b-it"
    
    logger.info(f"Auto-selected model: {model}")
    
    return ParaphraseGenerator(
        model_name=model,
        device=device,
        load_in_4bit=quantize
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with mock generator
    from .models import Question, TaskType
    
    question = Question(
        id="test_1",
        text="What is the capital of France?",
        answer="Paris",
        task_type=TaskType.COMMONSENSE,
        source_dataset="test"
    )
    
    print("Testing with mock generator...")
    generator = ParaphraseGenerator(use_mock=True)
    paraphrases = generator.generate_paraphrases_for_question(question, num_paraphrases=4)
    
    print(f"\nOriginal: {question.text}")
    for p in paraphrases:
        print(f"  [{p.paraphrase_type.value}] {p.text}")
    
    print("\n" + "="*50)
    print("To use with real models, run:")
    print("  generator = ParaphraseGenerator(model_name='Qwen/Qwen2.5-3B-Instruct')")
    print("  # Or for low memory:")
    print("  generator = ParaphraseGenerator(model_name='google/gemma-2-2b-it', load_in_4bit=True)")
