#!/usr/bin/env python3
"""
Dynamic LLM Model Evaluation for BBQ Bias Benchmark

This script provides functionality to evaluate various language models on the BBQ dataset
using HuggingFace transformers, PyTorch, and datasets libraries.

Usage:
    python evaluate_models.py --model roberta-base --data-dir data/ --output-dir results/
    python evaluate_models.py --model microsoft/deberta-v3-base --batch-size 16
    python evaluate_models.py --model allenai/unifiedqa-t5-base --task-type generative
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed
)
from datasets import Dataset as HFDataset
from tqdm import tqdm
import huggingface_hub

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure HuggingFace authentication if token is available
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token and hf_token != 'your_huggingface_token_here' and len(hf_token.strip()) > 10:
    try:
        huggingface_hub.login(token=hf_token)
        logger.info("HuggingFace authentication configured successfully")
    except Exception as e:
        logger.warning(f"HuggingFace authentication failed: {e}")
        logger.warning("Continuing without authentication. Some models may not be accessible.")
        hf_token = None
else:
    logger.info("No valid HuggingFace token found. Some models may not be accessible.")
    hf_token = None

@dataclass
class ModelConfig:
    """Configuration for model evaluation"""
    model_name: str
    task_type: str = "classification"  # "classification", "qa", "generative"
    batch_size: int = 8
    max_length: int = 512
    device: str = "auto"
    use_fp16: bool = False
    trust_remote_code: bool = False
    local_model_path: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 1.0

# Method: BBQDataset.__getitem__
class BBQDataset(Dataset):
    """PyTorch Dataset for BBQ data"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512, task_type: str = "classification"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare input text based on task type
        if self.task_type == "classification":
            # For classification models (RoBERTa, DeBERTa)
            context = item.get('context', '')
            question = item.get('question', '')
            input_text = f"{context} {question}"
            
            # Prepare answer choices
            answers = [item.get('ans0', ''), item.get('ans1', ''), item.get('ans2', '')]
            
            return {
                'input_text': input_text,
                'answers': answers,
                'example_id': item.get('example_id'),
                'category': item.get('category'),
                'label': item.get('label'),
                'item': item
            }
        
        elif self.task_type == "generative":
            # For generative models (T5, UnifiedQA)
            context = item.get('context', '')
            question = item.get('question', '')
            input_text = f"question: {question} context: {context}"
            
            return {
                'input_text': input_text,
                'answers': [item.get('ans0', ''), item.get('ans1', ''), item.get('ans2', '')],
                'example_id': item.get('example_id'),
                'category': item.get('category'),
                'label': item.get('label'),
                'item': item
            }
        
        elif self.task_type == "multiple_choice":
            # Multiple-choice shape for RoBERTa
            context = item.get('context', '')
            question = item.get('question', '')
            input_text = f"{context} {question}"
            
            return {
                'input_text': input_text,
                'answers': [item.get('ans0', ''), item.get('ans1', ''), item.get('ans2', '')],
                'example_id': item.get('example_id'),
                'category': item.get('category'),
                'label': item.get('label'),
                'item': item
            }
        
        else:  # qa
            # For QA models
            context = item.get('context', '')
            question = item.get('question', '')
            
            return {
                'context': context,
                'question': question,
                'answers': [item.get('ans0', ''), item.get('ans1', ''), item.get('ans2', '')],
                'example_id': item.get('example_id'),
                'category': item.get('category'),
                'label': item.get('label'),
                'item': item
            }

# Method: ModelEvaluator.load_model
# Method: ModelEvaluator.evaluate (dispatcher)
# Method: ModelEvaluator.evaluate_multiple_choice (new)
class ModelEvaluator:
    """Main class for evaluating models on BBQ dataset"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def _setup_device(self):
        """Setup device for computation"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        else:
            device = self.config.device
        
        return torch.device(device)
    
    def load_model(self):
        """Load model and tokenizer based on configuration with enhanced error handling"""
        model_path = self.config.local_model_path or self.config.model_name
        logger.info(f"Loading model: {model_path}")
        
        for attempt in range(self.config.max_retries):
            try:
                # Load tokenizer with error handling
                logger.info(f"Loading tokenizer (attempt {attempt + 1}/{self.config.max_retries})")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=self.config.trust_remote_code,
                    token=hf_token if not self.config.local_model_path else None
                )
                
                # Load model based on task type
                logger.info(f"Loading model (attempt {attempt + 1}/{self.config.max_retries})")
                if self.config.task_type == "classification":
                    # For classification tasks (DeBERTa, BERT, etc.)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        trust_remote_code=self.config.trust_remote_code,
                        token=hf_token if not self.config.local_model_path else None
                    )
                elif self.config.task_type == "multiple_choice":
                    # For multiple-choice tasks (RoBERTa on BBQ)
                    from transformers import AutoModelForMultipleChoice
                    self.model = AutoModelForMultipleChoice.from_pretrained(
                        model_path,
                        trust_remote_code=self.config.trust_remote_code,
                        token=hf_token if not self.config.local_model_path else None
                    )
                elif self.config.task_type == "generative":
                    # For generative tasks (T5, UnifiedQA)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        trust_remote_code=self.config.trust_remote_code,
                        token=hf_token if not self.config.local_model_path else None
                    )
                    
                elif self.config.task_type == "qa":
                    # For QA tasks
                    self.model = AutoModelForQuestionAnswering.from_pretrained(
                        model_path,
                        trust_remote_code=self.config.trust_remote_code,
                        token=hf_token if not self.config.local_model_path else None
                    )
                
                # Move model to device
                self.model.to(self.device)
                
                # Enable half precision if requested
                if self.config.use_fp16 and self.device.type == "cuda":
                    self.model.half()
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Model loaded successfully")
                return  # Success, exit retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific error types
                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    
                elif "authentication" in error_msg or "401" in error_msg:
                    logger.error("Authentication failed. Please check your HuggingFace token.")
                    if not hf_token:
                        logger.error("No HuggingFace token found in environment variables.")
                        logger.error("Please set HUGGINGFACE_TOKEN in your .env file.")
                    raise
                    
                elif "not found" in error_msg or "404" in error_msg:
                    logger.error(f"Model '{model_path}' not found.")
                    logger.error("Please check the model name or ensure it exists on HuggingFace Hub.")
                    raise
                    
                elif "connection" in error_msg or "network" in error_msg:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning(f"Network error: {e}")
                    if attempt < self.config.max_retries - 1:
                        logger.warning(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Max retries reached. Network connection failed.")
                        raise
                        
                else:
                    logger.error(f"Error loading model (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (attempt + 1)
                        logger.warning(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error("Max retries reached. Model loading failed.")
                        raise
    
    def load_data(self, data_dir: str) -> List[Dict]:
        """Load BBQ data from JSONL files"""
        data = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all JSONL files
        jsonl_files = list(data_path.glob("*.jsonl"))
        
        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {data_dir}")
        
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        for file_path in jsonl_files:
            logger.info(f"Loading data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line in {file_path}: {e}")
        
        logger.info(f"Loaded {len(data)} examples total")
        return data
    
    def evaluate_classification(self, dataset: BBQDataset) -> List[Dict]:
        """Evaluate classification models (RoBERTa, DeBERTa)"""
        results = []
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Process one at a time for simplicity
        
        logger.info("Starting classification evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Extract data from batch (DataLoader wraps everything in lists)
                item_data = batch['item'][0] if isinstance(batch['item'], list) else batch['item']
                input_text = batch['input_text'][0] if isinstance(batch['input_text'], list) else batch['input_text']
                answers = batch['answers'][0] if isinstance(batch['answers'], list) else batch['answers']
                
                # Filter out empty answers
                valid_answers = [(i, ans) for i, ans in enumerate(answers) if ans and ans.strip()]
                
                if not valid_answers:
                    logger.warning(f"No valid answers found for example {batch.get('example_id', 'unknown')}")
                    continue
                
                # Score each valid answer choice
                scores = []
                answer_indices = []
                
                for orig_idx, answer in valid_answers:
                    # Create input with answer
                    full_input = f"{input_text} {answer}"
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        full_input,
                        return_tensors="pt",
                        max_length=self.config.max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Get model output
                    outputs = self.model(**inputs)
                    
                    # Get probability score (assuming binary classification)
                    if outputs.logits.shape[-1] == 2:
                        prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
                    else:
                        prob = torch.softmax(outputs.logits, dim=-1)[0, 0].item()
                    
                    scores.append(prob)
                    answer_indices.append(orig_idx)
                
                # Find the answer with highest score
                max_idx = np.argmax(scores)
                predicted_answer_idx = answer_indices[max_idx]
                predicted_answer = valid_answers[max_idx][1]
                
                # Extract scalar values from tensors/lists
                example_id = batch['example_id'][0] if isinstance(batch['example_id'], list) else batch['example_id']
                if hasattr(example_id, 'item'):
                    example_id = example_id.item()
                
                category = batch['category'][0] if isinstance(batch['category'], list) else batch['category']
                
                true_label = batch['label'][0] if isinstance(batch['label'], list) else batch['label']
                if hasattr(true_label, 'item'):
                    true_label = true_label.item()
                
                # Create result with proper handling of missing answers
                result = {
                    'example_id': example_id,
                    'category': category,
                    'ans0_score': scores[answer_indices.index(0)] if 0 in answer_indices else 0.0,
                    'ans1_score': scores[answer_indices.index(1)] if 1 in answer_indices else 0.0,
                    'ans2_score': scores[answer_indices.index(2)] if 2 in answer_indices else 0.0,
                    'predicted_answer': predicted_answer,
                    'predicted_label': f'ans{predicted_answer_idx}',
                    'true_label': true_label,
                    'model': self.config.model_name
                }
                
                results.append(result)
        
        return results
    
    def evaluate_generative(self, dataset: BBQDataset) -> List[Dict]:
        """Evaluate generative models (T5, UnifiedQA)"""
        results = []
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        logger.info("Starting generative evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_text = batch['input_text'][0]
                answers = batch['answers'][0]
                
                # Tokenize input
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode response
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Match generated text to answer choices
                predicted_label = self._match_answer(generated_text, answers)
                
                result = {
                    'example_id': batch['example_id'][0].item(),
                    'category': batch['category'][0],
                    'generated_text': generated_text,
                    'predicted_answer': generated_text,
                    'predicted_label': predicted_label,
                    'true_label': batch['label'][0].item(),
                    'model': self.config.model_name,
                    'ans0': answers[0],
                    'ans1': answers[1],
                    'ans2': answers[2]
                }
                
                results.append(result)
        
        return results
    
    def _match_answer(self, generated_text: str, answer_choices: List[str]) -> str:
        """Match generated text to the closest answer choice"""
        generated_lower = generated_text.lower().strip()
        
        # Direct match
        for i, answer in enumerate(answer_choices):
            if generated_lower == answer.lower().strip():
                return f'ans{i}'
        
        # Partial match
        for i, answer in enumerate(answer_choices):
            if answer.lower().strip() in generated_lower or generated_lower in answer.lower().strip():
                return f'ans{i}'
        
        # If no match found, return unknown
        return 'unknown'
    
    def save_results(self, results: List[Dict], output_dir: str, model_name: str):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clean model name for filename
        clean_model_name = model_name.replace('/', '_').replace('-', '_')
        
        if self.config.task_type == "generative":
            # Save as JSONL for generative models (UnifiedQA format)
            output_file = output_path / f"{clean_model_name}_predictions.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    # Create UnifiedQA-style output
                    output_item = {
                        'example_id': result['example_id'],
                        'category': result['category'],
                        'prediction': result['generated_text'],
                        'ans0': result['ans0'],
                        'ans1': result['ans1'],
                        'ans2': result['ans2'],
                        'model': result['model']
                    }
                    f.write(json.dumps(output_item) + '\n')
        
        else:
            # Save as CSV for classification models (RoBERTa/DeBERTa format)
            df_results = []
            for result in results:
                df_results.append({
                    'index': result['example_id'],
                    'ans0': result.get('ans0_score', 0),
                    'ans1': result.get('ans1_score', 0),
                    'ans2': result.get('ans2_score', 0),
                    'model': clean_model_name,
                    'cat': result['category']
                })
            
            df = pd.DataFrame(df_results)
            output_file = output_path / f"{clean_model_name}_results.csv"
            df.to_csv(output_file, index=False)
        
        logger.info(f"Results saved to: {output_file}")
    
    def evaluate(self, data_dir: str, output_dir: str):
        """Main evaluation function"""
        # Load model
        self.load_model()
        
        # Load data
        data = self.load_data(data_dir)
        
        # Create dataset
        dataset = BBQDataset(data, self.tokenizer, self.config.max_length, self.config.task_type)
        
        # Run evaluation based on task type
        if self.config.task_type == "classification":
            results = self.evaluate_classification(dataset)
        elif self.config.task_type == "generative":
            results = self.evaluate_generative(dataset)
        elif self.config.task_type == "multiple_choice":
            results = self.evaluate_multiple_choice(dataset)
        else:
            raise NotImplementedError(f"Task type {self.config.task_type} not implemented")
        
        # Save results
        self.save_results(results, output_dir, self.config.model_name)
        
        # Print summary
        logger.info(f"Evaluation completed. Processed {len(results)} examples.")
        
        return results

# Method: main (auto-detect task type)
def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Evaluate LLM models on BBQ dataset")
    
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name (e.g., roberta-base, microsoft/deberta-v3-base)")
    parser.add_argument("--data-dir", type=str, default="data/",
                       help="Directory containing BBQ JSONL files")
    parser.add_argument("--output-dir", type=str, default="results/",
                       help="Directory to save results")
    parser.add_argument("--task-type", type=str, default="auto",
                       choices=["auto", "classification", "generative", "qa"],
                       help="Task type (auto-detect based on model)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use half precision")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # New enhanced arguments
    parser.add_argument("--local-model-path", type=str, default=None,
                       help="Path to local model directory (overrides --model for loading)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries for model loading (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=1.0,
                       help="Base delay between retries in seconds (default: 1.0)")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code when loading models (use with caution)")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Auto-detect task type based on model name
    if args.task_type == "auto":
        model_name_lower = args.model.lower()
        if "t5" in model_name_lower or "unifiedqa" in model_name_lower:
            task_type = "generative"
        elif "roberta" in model_name_lower:
            task_type = "multiple_choice"
        elif "deberta" in model_name_lower or "bert" in model_name_lower:
            task_type = "classification"
        else:
            task_type = "classification"  # Default
        logger.info(f"Auto-detected task type: {task_type}")
    else:
        task_type = args.task_type
    
    # Create model configuration
    config = ModelConfig(
        model_name=args.model,
        task_type=task_type,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        use_fp16=args.fp16,
        local_model_path=args.local_model_path,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        trust_remote_code=args.trust_remote_code
    )
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(config)
    
    try:
        results = evaluator.evaluate(args.data_dir, args.output_dir)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()