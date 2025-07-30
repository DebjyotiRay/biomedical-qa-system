#!/usr/bin/env python3
"""
Training script for Biomedical QA System
Fine-tunes the model using LoRA on biomedical question-answer pairs
"""

import json
import os
import logging
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

from src.model import BiomedicalModel
from src.utils import load_config, load_environment_variables

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BiomedicalQADataset(Dataset):
    """Dataset class for biomedical Q&A pairs"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Created dataset with {len(data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create a prompt similar to what the model expects during inference
        question = item['question']
        answer = item['answer']
        
        # Use the same prompt format as inference with examples
        prompt = f"""Question: Which genes are commonly mutated in breast cancer?
Answer: • BRCA1
• BRCA2
• TP53
• PIK3CA
• PTEN
• GATA3
• CDH1

Question: Which proteins are involved in DNA repair?
Answer: • BRCA1
• BRCA2
• RAD51
• ATM
• ATR
• MSH2
• MLH1

Question: Which drugs are EGFR inhibitors?
Answer: • Erlotinib
• Gefitinib
• Afatinib
• Osimertinib
• Dacomitinib

Question: {question}
Answer: {answer}"""
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For causal LM, labels = input_ids
        }

def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training examples from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Training data file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing training data: {e}")
        raise

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """Split data into train and validation sets"""
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data

def main():
    """Main training function"""
    logger.info("Starting biomedical QA model training...")
    
    # Load environment variables and configuration
    load_environment_variables()
    config = load_config("config.json")
    
    # Create output directory
    output_dir = config.get("output", {}).get("dir", "output")
    peft_output_dir = os.path.join(output_dir, "peft_adapter")
    os.makedirs(peft_output_dir, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing model...")
    model = BiomedicalModel(config)
    
    # Load base model
    if not model.load_model():
        logger.error("Failed to load base model")
        return False
    
    # Apply PEFT (LoRA)
    if not model.apply_peft():
        logger.error("Failed to apply PEFT")
        return False
    
    # Load training data
    training_data_path = "data/training_data.json"
    data = load_training_data(training_data_path)
    
    # Split data
    train_data, val_data = split_data(data, train_ratio=0.8)
    
    # Create datasets
    max_length = config.get("model", {}).get("max_length", 512)
    train_dataset = BiomedicalQADataset(train_data, model.tokenizer, max_length)
    val_dataset = BiomedicalQADataset(val_data, model.tokenizer, max_length) if val_data else None
    
    # Training arguments
    training_config = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=peft_output_dir,
        per_device_train_batch_size=training_config.get("batch_size", 2),
        per_device_eval_batch_size=training_config.get("batch_size", 2),
        learning_rate=training_config.get("learning_rate", 5e-5),
        num_train_epochs=training_config.get("num_epochs", 3),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        eval_strategy="steps" if val_dataset else "no",  # Changed from evaluation_strategy
        eval_steps=training_config.get("eval_steps", 100) if val_dataset else None,
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=3,
        logging_steps=50,
        warmup_steps=100,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        seed=training_config.get("seed", 42),
        dataloader_pin_memory=False,  # Can help with performance
        remove_unused_columns=False,
        report_to=[],  # Disable wandb/tensorboard logging
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=model.tokenizer,
        mlm=False,  # We're doing causal language modeling, not masked LM
    )
    
    # Callbacks
    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
    
    # Initialize trainer
    trainer = Trainer(
        model=model.peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Print model info
    logger.info("Model training parameters:")
    model.peft_model.print_trainable_parameters()
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save the final model
        logger.info(f"Saving PEFT adapter to {peft_output_dir}")
        model.save_peft_adapter(peft_output_dir)
        
        # Save training metrics
        if trainer.state.log_history:
            metrics_file = os.path.join(peft_output_dir, "training_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(trainer.state.log_history, f, indent=2)
            logger.info(f"Training metrics saved to {metrics_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Training process completed successfully!")
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        print("Your fine-tuned model adapter has been saved to: output/peft_adapter/")
        print("You can now use it for inference by loading the adapter.")
        print("="*50)
    else:
        logger.error("Training process failed!")
        exit(1)
