"""
Adapted from: 
https://huggingface.co/blog/train-sentence-transformers#loss-function
"""
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)

# Load a model to train/finetune
model = SentenceTransformer("intfloat/multilingual-e5-large")

# Initialize the CoSENTLoss
# This loss requires pairs of text and a floating point similarity score as a label
loss = TripletLoss(model=model)

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset("DDSC/da-wikipedia-queries-gemma-processed")
train_dataset = train_dataset.remove_columns(["negative_index_pos"])

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/multilingual-e5-large-ddsc",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if your GPU can't handle FP16
    bf16=True,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="test",  # Used in W&B if `wandb` is installed
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

model.save_pretrained("models/multilingual-e5-large-ddsc/final")

