# train_model.py
"""
Fine-tune a T5 model for legal text generation.
Compatible with transformers 5.12.1 and torch 2.12.1
"""

import json
import torch
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os
from tqdm import tqdm

def load_data(json_path: str):
    """Load preprocessed data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def train_model(
    json_data_path: str,
    model_name: str = 't5-small',
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    max_length: int = 512,
    output_dir: str = './fine_tuned_model'
):
    """
    Fine-tune a T5 model for legal text generation.
    Compatible with transformers 5.12.1
    """
    # Load data
    print("Loading data...")
    data = load_data(json_data_path)
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Prepare dataset
    def preprocess_function(examples):
        # Tokenize inputs
        inputs = tokenizer(
            examples['input'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Tokenize targets
        targets = tokenizer(
            examples['target'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    # Split data
    from data_to_json import split_data
    train_data, test_data = split_data(data)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'input': train_data['inputs'],
        'target': train_data['targets']
    })
    
    test_dataset = Dataset.from_dict({
        'input': test_data['inputs'],
        'target': test_data['targets']
    })
    
    # Preprocess datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=max_length,
        label_pad_token_id=tokenizer.pad_token_id
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=50,
        report_to='none',
        save_total_limit=2,
        push_to_hub=False,
    )
    
    # Initialize trainer - REMOVED tokenizer parameter
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate
    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test loss: {eval_results['eval_loss']:.4f}")
    
    return trainer, model, tokenizer

def train_mini_model(
    json_data_path: str,
    output_dir: str = './fine_tuned_mini_model'
):
    """
    Train a mini T5 model for faster experimentation.
    """
    model_name = 't5-small'
    
    return train_model(
        json_data_path=json_data_path,
        model_name=model_name,
        num_epochs=3,
        batch_size=4,
        learning_rate=3e-4,
        output_dir=output_dir
    )

if __name__ == "__main__":
    print("Training mini T5 model...")
    trainer, model, tokenizer = train_mini_model(
        json_data_path='legal_data.json',
        output_dir='./fine_tuned_mini_model'
    )