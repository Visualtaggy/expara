"""
ExPara Training Module
Methods for improving explanation consistency through training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for consistency-aware training."""
    # Model
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Training params
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_length: int = 1024
    
    # Loss weights
    lm_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.5
    
    # DPO params
    dpo_beta: float = 0.1
    
    # Contrastive params
    contrastive_temperature: float = 0.07
    
    # Output
    output_dir: str = "models/"
    save_steps: int = 500


class ConsistencyDataset(Dataset):
    """Dataset for consistency-aware training."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024
    ):
        """
        Args:
            data: List of dicts with keys:
                - question: original question
                - paraphrase: paraphrased question
                - explanation: target explanation
                - explanation_alt: alternative explanation (for consistency)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        # Format prompts
        prompt_original = self._format_prompt(item["question"])
        prompt_paraphrase = self._format_prompt(item["paraphrase"])
        
        # Tokenize
        tokens_original = self.tokenizer(
            prompt_original + item["explanation"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        tokens_paraphrase = self.tokenizer(
            prompt_paraphrase + item["explanation"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids_original": tokens_original["input_ids"].squeeze(),
            "attention_mask_original": tokens_original["attention_mask"].squeeze(),
            "input_ids_paraphrase": tokens_paraphrase["input_ids"].squeeze(),
            "attention_mask_paraphrase": tokens_paraphrase["attention_mask"].squeeze(),
            "prompt_length_original": len(self.tokenizer(prompt_original)["input_ids"]),
            "prompt_length_paraphrase": len(self.tokenizer(prompt_paraphrase)["input_ids"])
        }
    
    def _format_prompt(self, question: str) -> str:
        return f"""Answer the following question. Think step by step and show your reasoning.

Question: {question}

Let's think step by step:
"""


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024
    ):
        """
        Args:
            data: List of dicts with keys:
                - question: the question
                - chosen: preferred explanation (consistent)
                - rejected: non-preferred explanation (inconsistent)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]
        
        prompt = f"""Answer the following question. Think step by step and show your reasoning.

Question: {item["question"]}

Let's think step by step:
"""
        
        # Tokenize chosen
        chosen_full = prompt + item["chosen"]
        chosen_tokens = self.tokenizer(
            chosen_full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_full = prompt + item["rejected"]
        rejected_tokens = self.tokenizer(
            rejected_full,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        prompt_length = len(self.tokenizer(prompt)["input_ids"])
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(),
            "prompt_length": prompt_length
        }


class ConsistencyTrainer:
    """
    Trainer for improving explanation consistency.
    Uses a combination of:
    1. Standard language modeling loss
    2. Consistency regularization (similar explanations for paraphrased inputs)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: TrainingConfig,
        embedding_model=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.embedding_model = embedding_model
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def compute_lm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Compute language modeling loss (only on response tokens)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        # Mask out prompt tokens from loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Create loss mask
        loss_mask = torch.ones_like(shift_labels, dtype=torch.float)
        loss_mask[:, :prompt_length-1] = 0
        
        # Compute per-token loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss = loss.view(shift_labels.size())
        
        # Apply mask and average
        masked_loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        return masked_loss
    
    def compute_consistency_loss(
        self,
        hidden_original: torch.Tensor,
        hidden_paraphrase: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between explanations for original and paraphrased inputs.
        Uses cosine similarity loss to encourage similar hidden representations.
        """
        # Mean pool over sequence
        emb_original = hidden_original.mean(dim=1)
        emb_paraphrase = hidden_paraphrase.mean(dim=1)
        
        # Normalize
        emb_original = F.normalize(emb_original, p=2, dim=-1)
        emb_paraphrase = F.normalize(emb_paraphrase, p=2, dim=-1)
        
        # Cosine similarity loss (want to maximize similarity -> minimize 1 - sim)
        similarity = (emb_original * emb_paraphrase).sum(dim=-1)
        loss = (1 - similarity).mean()
        
        return loss
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Get hidden states for original
        outputs_original = self.model(
            input_ids=batch["input_ids_original"],
            attention_mask=batch["attention_mask_original"],
            output_hidden_states=True
        )
        
        # Get hidden states for paraphrase
        outputs_paraphrase = self.model(
            input_ids=batch["input_ids_paraphrase"],
            attention_mask=batch["attention_mask_paraphrase"],
            output_hidden_states=True
        )
        
        # Language modeling loss (both should produce good explanations)
        lm_loss_original = self.compute_lm_loss(
            batch["input_ids_original"],
            batch["attention_mask_original"],
            batch["prompt_length_original"]
        )
        
        lm_loss_paraphrase = self.compute_lm_loss(
            batch["input_ids_paraphrase"],
            batch["attention_mask_paraphrase"],
            batch["prompt_length_paraphrase"]
        )
        
        lm_loss = (lm_loss_original + lm_loss_paraphrase) / 2
        
        # Consistency loss
        hidden_original = outputs_original.hidden_states[-1]
        hidden_paraphrase = outputs_paraphrase.hidden_states[-1]
        consistency_loss = self.compute_consistency_loss(
            hidden_original, hidden_paraphrase
        )
        
        # Total loss
        total_loss = (
            self.config.lm_loss_weight * lm_loss +
            self.config.consistency_loss_weight * consistency_loss
        )
        
        # Backward
        total_loss.backward()
        
        return {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "consistency_loss": consistency_loss.item()
        }
    
    def train(self, dataset: ConsistencyDataset):
        """Full training loop."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        num_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(num_steps * self.config.warmup_ratio)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        global_step = 0
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                losses = self.train_step(batch)
                epoch_losses.append(losses)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{losses['total_loss']:.4f}",
                    "lm": f"{losses['lm_loss']:.4f}",
                    "cons": f"{losses['consistency_loss']:.4f}"
                })
                
                # Save checkpoint
                if global_step > 0 and global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)
            
            # Epoch summary
            avg_losses = {
                k: sum(l[k] for l in epoch_losses) / len(epoch_losses)
                for k in epoch_losses[0].keys()
            }
            logger.info(f"Epoch {epoch+1} - Avg losses: {avg_losses}")
        
        # Save final model
        self._save_checkpoint("final")
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function with padding."""
        # Find max lengths
        max_len_orig = max(len(item["input_ids_original"]) for item in batch)
        max_len_para = max(len(item["input_ids_paraphrase"]) for item in batch)
        
        # Pad
        padded = {
            "input_ids_original": [],
            "attention_mask_original": [],
            "input_ids_paraphrase": [],
            "attention_mask_paraphrase": [],
            "prompt_length_original": [],
            "prompt_length_paraphrase": []
        }
        
        for item in batch:
            # Pad original
            pad_len = max_len_orig - len(item["input_ids_original"])
            padded["input_ids_original"].append(
                F.pad(item["input_ids_original"], (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            padded["attention_mask_original"].append(
                F.pad(item["attention_mask_original"], (0, pad_len), value=0)
            )
            
            # Pad paraphrase
            pad_len = max_len_para - len(item["input_ids_paraphrase"])
            padded["input_ids_paraphrase"].append(
                F.pad(item["input_ids_paraphrase"], (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            padded["attention_mask_paraphrase"].append(
                F.pad(item["attention_mask_paraphrase"], (0, pad_len), value=0)
            )
            
            padded["prompt_length_original"].append(item["prompt_length_original"])
            padded["prompt_length_paraphrase"].append(item["prompt_length_paraphrase"])
        
        return {
            k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v
            for k, v in padded.items()
        }
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved checkpoint to {path}")


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    Trains model to prefer consistent explanations over inconsistent ones.
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: TrainingConfig
    ):
        self.model = model
        self.ref_model = ref_model  # Frozen reference model
        self.tokenizer = tokenizer
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """Compute log probabilities for response tokens."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask prompt tokens
        mask = torch.ones_like(token_log_probs)
        mask[:, :prompt_length-1] = 0
        mask = mask * attention_mask[:, 1:]
        
        # Sum log probs
        return (token_log_probs * mask).sum(dim=-1)
    
    def dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """Compute DPO loss."""
        # Log ratios
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        
        # DPO loss
        losses = -F.logsigmoid(
            self.config.dpo_beta * (chosen_logratios - rejected_logratios)
        )
        
        return losses.mean()
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single DPO training step."""
        self.model.train()
        
        # Policy log probs
        chosen_logps = self.compute_log_probs(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["prompt_length"]
        )
        
        rejected_logps = self.compute_log_probs(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["prompt_length"]
        )
        
        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["prompt_length"]
            )
            
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["prompt_length"]
            )
        
        # Compute loss
        loss = self.dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )
        
        loss.backward()
        
        # Metrics
        with torch.no_grad():
            chosen_rewards = self.config.dpo_beta * (chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.dpo_beta * (rejected_logps - ref_rejected_logps)
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "accuracy": accuracy.item()
        }
    
    def train(self, dataset: DPODataset):
        """Full DPO training loop."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        global_step = 0
        for epoch in range(self.config.num_epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in pbar:
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                metrics = self.train_step(batch)
                
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.3f}"
                })
                
                global_step += 1
        
        # Save
        self.model.save_pretrained(
            os.path.join(self.config.output_dir, "dpo_final")
        )


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Available trainers:")
    print("  - ConsistencyTrainer: Consistency-aware fine-tuning")
    print("  - DPOTrainer: Direct Preference Optimization for consistency")
