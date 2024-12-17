import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk, Dataset
import torch
import evaluate
import nltk
import numpy as np
import random
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def subsample_dataset(dataset, num_samples, seed=42):
    """Randomly subsample the dataset to the specified size"""
    if len(dataset) <= num_samples:
        return dataset
    
    # Set seed for reproducibility
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)
    return Dataset.from_dict(dataset[indices])

def parse_args():
    parser = argparse.ArgumentParser()
    # Add local_rank argument for deepspeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")
    
    # Original arguments
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--repository_id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--generation_max_length", type=int, default=140)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True if torch.cuda.get_device_capability()[0] == 8 else False)
    parser.add_argument("--hf_token", type=str, default=HfFolder.get_token())
    parser.add_argument("--output_dir", type=str, default="./save/tmp")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--train_samples", type=int, default=-1, help="Number of training samples to use. -1 means use all data.")
    
    args = parser.parse_known_args()[0]
    return args

def training_function(args):
    set_seed(args.seed)

    # Load dataset and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset_path, "eval"))
    
    # Subsample training data if specified
    if args.train_samples > 0:
        print(f"Subsampling training dataset from {len(train_dataset)} to {args.train_samples} examples...")
        train_dataset = subsample_dataset(train_dataset, args.train_samples, args.seed)
        print(f"New training dataset size: {len(train_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,
    )
    
    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=False,
        bf16=args.bf16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="wandb",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
        local_rank=args.local_rank,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    torch.cuda.empty_cache()
    trainer.train()

    # Save tokenizer and create model card
    if args.local_rank in [-1, 0]:  # Only save on main process
        tokenizer.save_pretrained(args.output_dir)
        trainer.create_model_card()

def main():
    args = parse_args()
    training_function(args)

if __name__ == "__main__":
    main()