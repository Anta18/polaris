"""Training script for LoRA adapter on clickbait detection task."""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import json

# Configuration
BASE_MODEL_PATH = "models/base_model"
LORA_ADAPTER_PATH = "models/lora_adapter"
MAX_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def load_and_prepare_data(train_file=None, val_file=None, test_file=None, dataset_file=None, split_ratio=0.2):
    """
    Load and prepare dataset for training.
    
    Args:
        train_file: Path to training CSV/JSON file
        val_file: Path to validation CSV/JSON file  
        test_file: Path to test CSV/JSON file
        dataset_file: Path to full dataset CSV file (will be split)
        split_ratio: Ratio for train/val/test split (default: 0.2 for val, 0.2 for test)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from sklearn.model_selection import train_test_split
    
    
    if dataset_file and dataset_file.startswith("hf://"):
        # Format: hf://dataset_name
        dataset_name = dataset_file.replace("hf://", "")
        print(f"Loading dataset from HuggingFace: {dataset_name}")
        try:
            ds = load_dataset(dataset_name)
            if "train" in ds:
                dataset = ds["train"]
            else:
                dataset = list(ds.values())[0]
            
            
            rows = []
            for item in dataset:
                title = item.get("title", item.get("text", ""))
                content = item.get("content", item.get("body", ""))
                
                label = item.get("label", item.get("clickbait", item.get("clickbait_label", 0)))
                label = int(label) if label not in [0, 1] else label
                
                rows.append({
                    "title": str(title).strip() if title else "",
                    "content": str(content).strip() if content else "",
                    "label": label
                })
            
            df = pd.DataFrame(rows)
            print(f"Loaded {len(df)} samples from HuggingFace")
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            return None, None, None
    
    
    elif dataset_file and os.path.exists(dataset_file):
        print(f"Loading dataset from {dataset_file}")
        df = pd.read_csv(dataset_file)
        print(f"Loaded {len(df)} samples")
        
        
        column_mapping = {
            "title": ["title", "headline", "text"],
            "content": ["content", "body", "article", "text_body"],
            "label": ["label", "clickbait", "clickbait_label", "is_clickbait", "target"]
        }
        
        for standard_name, possible_names in column_mapping.items():
            if standard_name not in df.columns:
                for name in possible_names:
                    if name in df.columns:
                        df.rename(columns={name: standard_name}, inplace=True)
                        break
        
     
        if "title" not in df.columns:
            raise ValueError("Dataset must have a 'title' column")
        if "label" not in df.columns:
            raise ValueError("Dataset must have a 'label' column")
        

        if "content" not in df.columns:
            df["content"] = ""  # Empty content
            print(" No 'content' column found. Using title only.")
        
        # Clean data
        df = df.dropna(subset=["title", "label"])
        df["title"] = df["title"].astype(str)
        df["content"] = df["content"].fillna("").astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        
        
        df["label"] = df["label"].apply(lambda x: 1 if x > 0 else 0)
        
        print(f"Cleaned dataset: {len(df)} samples")
        print(f"Clickbait ratio: {df['label'].mean():.2%}")
        
        
        if not train_file:
            train_df, temp_df = train_test_split(
                df, 
                test_size=split_ratio * 2, 
                random_state=42, 
                stratify=df["label"]
            )
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=0.5, 
                random_state=42, 
                stratify=temp_df["label"]
            )
            
            print(f"Split dataset:")
            print(f"  Train: {len(train_df)} samples ({train_df['label'].mean():.2%} clickbait)")
            print(f"  Val: {len(val_df)} samples ({val_df['label'].mean():.2%} clickbait)")
            print(f"  Test: {len(test_df)} samples ({test_df['label'].mean():.2%} clickbait)")
            
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            test_dataset = Dataset.from_pandas(test_df)
            
            return train_dataset, val_dataset, test_dataset
    
    
    if train_file and os.path.exists(train_file):
        print(f"Loading data from {train_file}")
        train_df = pd.read_csv(train_file)
        
        # Handle missing content
        if "content" not in train_df.columns:
            train_df["content"] = ""
        
        # Clean data
        train_df = train_df.dropna(subset=["title", "label"])
        train_df["title"] = train_df["title"].astype(str)
        train_df["content"] = train_df["content"].fillna("").astype(str)
        train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").astype(int)
        train_df["label"] = train_df["label"].apply(lambda x: 1 if x > 0 else 0)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = None
        test_dataset = None
        
        if val_file and os.path.exists(val_file):
            val_df = pd.read_csv(val_file)
            if "content" not in val_df.columns:
                val_df["content"] = ""
            val_df = val_df.dropna(subset=["title", "label"])
            val_df["title"] = val_df["title"].astype(str)
            val_df["content"] = val_df["content"].fillna("").astype(str)
            val_df["label"] = pd.to_numeric(val_df["label"], errors="coerce").astype(int)
            val_df["label"] = val_df["label"].apply(lambda x: 1 if x > 0 else 0)
            val_dataset = Dataset.from_pandas(val_df)
        
        if test_file and os.path.exists(test_file):
            test_df = pd.read_csv(test_file)
            if "content" not in test_df.columns:
                test_df["content"] = ""
            test_df = test_df.dropna(subset=["title", "label"])
            test_df["title"] = test_df["title"].astype(str)
            test_df["content"] = test_df["content"].fillna("").astype(str)
            test_df["label"] = pd.to_numeric(test_df["label"], errors="coerce").astype(int)
            test_df["label"] = test_df["label"].apply(lambda x: 1 if x > 0 else 0)
            test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, val_dataset, test_dataset
    
    
    print("No dataset file provided. Creating dummy dataset for testing...")
    print("Replace this with your actual dataset!")
    
    dummy_data = {
        "title": [
            "You Won't Believe What Happened Next!",
            "Scientists Discover New Breakthrough in Medicine",
            "This One Trick Will Change Your Life!",
            "Local News: City Council Meeting Scheduled",
            "SHOCKING: The Truth They Don't Want You to Know!",
            "Breaking: Major Policy Announcement Today"
        ],
        "content": [
            "Clickbait content here...",
            "Scientific article content...",
            "Clickbait trick content...",
            "Local news content...",
            "Sensational content...",
            "News article content..."
        ],
        "label": [1, 0, 1, 0, 1, 0]  # 1 = clickbait, 0 = clean
    }
    
    df = pd.DataFrame(dummy_data)
    dataset = Dataset.from_pandas(df)
    train_test = dataset.train_test_split(test_size=0.3)
    
    return train_test["train"], train_test["test"], None

def preprocess_function(examples, tokenizer):
    """Tokenize and prepare inputs for the model."""
    # Handle empty or missing content
    titles = examples.get("title", [])
    contents = examples.get("content", [""] * len(titles)) if "content" in examples else [""] * len(titles)
    
    
    texts = []
    for title, content in zip(titles, contents):
        title = str(title).strip() if title else ""
        content = str(content).strip() if content else ""
        
        if content:
            text = f"title: {title} [SEP] content: {content}"
        else:
            
            text = title
        
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    
    
    if "label" in examples:
        tokenized["labels"] = [float(label) for label in examples["label"]]
    
    return tokenized

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    
    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    labels = labels.astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, binary_predictions, average='binary', zero_division=0
    )
    
    # AUC-ROC
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def setup_lora_model(base_model_path, output_path):
    """Setup LoRA configuration and model."""
    print(f"Loading base model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,  # Binary classification
        problem_type="regression"  # Use regression for sigmoid output
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # LoRA rank 
        lora_alpha=32,  # LoRA alpha 
        lora_dropout=0.1,  # Dropout for LoRA layers
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # DistilBERT attention modules
        bias="none",  
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_adapter(
    train_dataset,
    val_dataset=None,
    base_model_path=BASE_MODEL_PATH,
    output_path=LORA_ADAPTER_PATH,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE
):
    """Train LoRA adapter."""
    
    # Setup model and tokenizer
    model, tokenizer = setup_lora_model(base_model_path, output_path)
    
    # Preprocess datasets
    print("Preprocessing training data...")
    
    columns_to_remove = [col for col in train_dataset.column_names 
                        if col not in ["input_ids", "attention_mask", "labels"]]
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=columns_to_remove if columns_to_remove else None
    )
    
    if val_dataset:
        print("Preprocessing validation data...")
        columns_to_remove = [col for col in val_dataset.column_names 
                            if col not in ["input_ids", "attention_mask", "labels"]]
        val_dataset = val_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=columns_to_remove if columns_to_remove else None
        )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="f1" if val_dataset else None,
        save_total_limit=2,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_dataset else None,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving LoRA adapter to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save training info
    training_info = {
        "base_model": base_model_path,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": MAX_LEN,
    }
    
    with open(os.path.join(output_path, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("✓ Training complete!")
    print(f"✓ Adapter saved to {output_path}")
    
    # Evaluate on test set if available
    if val_dataset:
        print("\nFinal evaluation:")
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA adapter for clickbait detection")
    parser.add_argument("--dataset_file", type=str, help="Path to full dataset CSV file (will be split automatically)")
    parser.add_argument("--train_file", type=str, help="Path to training CSV file")
    parser.add_argument("--val_file", type=str, help="Path to validation CSV file")
    parser.add_argument("--test_file", type=str, help="Path to test CSV file")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="Ratio for validation/test split (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=LORA_ADAPTER_PATH, help="Output directory for adapter")
    
    args = parser.parse_args()
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        dataset_file=args.dataset_file,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        split_ratio=args.split_ratio
    )
    
    if train_dataset is None:
        print("Failed to load dataset. Please check your dataset file.")
        return
    
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")
    print(f"{'='*60}\n")
    
    # Train
    train_adapter(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_path=args.output_dir
    )

if __name__ == "__main__":
    main()

