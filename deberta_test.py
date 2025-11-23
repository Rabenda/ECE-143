import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from pathlib import Path

# ---- Proxy ---
result = subprocess.run(
    'bash -c "source /etc/network_turbo && env | grep proxy"',
    shell=True,
    capture_output=True,
    text=True,
)
output = result.stdout
for line in output.splitlines():
    if "=" in line:
        var, value = line.split("=", 1)
        os.environ[var] = value

# --- Config ---
MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LENGTH = 2048
BATCH_SIZE = 3
GRAD_ACCUM_STEPS = 5
LEARNING_RATE = 1e-5
EPOCHS = 1
PROTOTYPE_FRAC = 1
TEST_SIZE = 0.1

GRADIENT_CHECKPOINTING = False
RESUME_FROM_CHECKPOINT = False

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
OUTPUT_DIR = "./llm_preference_model_smart"
TENSORBOARD_DIR = "./tf-logs"
RUN_NAME = "trunc_2048_run"


# --- Find latest checkpoint ---
def get_latest_checkpoint(output_dir):
    """Find the most recent checkpoint in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    checkpoints = [
        d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None

    checkpoints_with_steps = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.name.split("-")[1])
            checkpoints_with_steps.append((step, ckpt))
        except (IndexError, ValueError):
            continue

    if not checkpoints_with_steps:
        return None

    latest_checkpoint = max(checkpoints_with_steps, key=lambda x: x[0])[1]
    return str(latest_checkpoint)


# --- Data ---
print("Loading data...")
df_train = pd.read_csv(TRAIN_PATH, engine="python")
df_test = pd.read_csv(TEST_PATH, engine="python")

df_train["label"] = (
    df_train["winner_model_a"] * 0 + df_train["winner_model_b"] * 1 + df_train["winner_tie"] * 2
)

df_train = df_train.dropna(subset=["prompt", "response_a", "response_b"])
print(f"Full dataset size: {len(df_train)}")

if PROTOTYPE_FRAC < 1.0:
    print(f"Creating a {PROTOTYPE_FRAC * 100}% stratified prototype dataset...")
    _, df_train = train_test_split(
        df_train, test_size=PROTOTYPE_FRAC, random_state=42, stratify=df_train["label"]
    )
    print(f"Prototype dataset size: {len(df_train)}")


# --- Smart Truncation Dataset ---
class SmartTruncationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False, augment=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.augment = augment

        self.prompts = df["prompt"].values
        self.response_as = df["response_a"].values
        self.response_bs = df["response_b"].values
        if not self.is_test:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def _smart_truncate(self, text, max_tokens):
        """Truncates text to max_tokens keeping Head + Tail."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return tokens
        else:
            head_len = max_tokens // 2
            tail_len = max_tokens - head_len
            return tokens[:head_len] + tokens[-tail_len:]

    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        resp_a = str(self.response_as[idx])
        resp_b = str(self.response_bs[idx])

        if not self.is_test:
            label = self.labels[idx]

        if self.augment and torch.rand(1).item() > 0.5:
            resp_a, resp_b = resp_b, resp_a
            if not self.is_test:
                label = 0 if label == 1 else (1 if label == 0 else 2)

        prompt_budget = 512
        response_budget = (self.max_length - prompt_budget - 10) // 2
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)[:prompt_budget]
        resp_a_ids = self._smart_truncate(resp_a, response_budget)
        resp_b_ids = self._smart_truncate(resp_b, response_budget)

        input_ids = (
            [self.tokenizer.cls_token_id]
            + prompt_ids
            + [self.tokenizer.sep_token_id]
            + resp_a_ids
            + [self.tokenizer.sep_token_id]
            + resp_b_ids
            + [self.tokenizer.sep_token_id]
        )

        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        if self.is_test:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            }


# --- TensorBoard Callback ---
class AdvancedTensorBoardCallback(TrainerCallback):
    """
    Logs gradients and weights.
    Standard metrics (Loss/Acc) are handled automatically by Trainer(report_to='tensorboard').
    """

    def __init__(self, writer):
        self.writer = writer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 100 == 0 and model is not None:
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm**0.5
            self.writer.add_scalar("gradients/total_norm", total_norm, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()


# --- Check for checkpoint ---
checkpoint_to_resume = None
if RESUME_FROM_CHECKPOINT:
    checkpoint_to_resume = get_latest_checkpoint(OUTPUT_DIR)
    if checkpoint_to_resume:
        print(f"Found checkpoint to resume from: {checkpoint_to_resume}")
    else:
        print("No checkpoint found. Starting training from scratch.")

# --- Model and Tokenizer ---
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Initialize Writer
tensorboard_log_dir = os.path.join(TENSORBOARD_DIR, RUN_NAME)
writer = SummaryWriter(log_dir=tensorboard_log_dir)

# --- Trainer ---
print("Splitting data...")
train_df, val_df = train_test_split(
    df_train, test_size=TEST_SIZE, random_state=42, stratify=df_train["label"]
)

# Use SmartTruncationDataset
train_dataset = SmartTruncationDataset(train_df, tokenizer, MAX_LENGTH, augment=True)
val_dataset = SmartTruncationDataset(val_df, tokenizer, MAX_LENGTH, augment=False)
test_dataset = SmartTruncationDataset(df_test, tokenizer, MAX_LENGTH, is_test=True)

print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    per_class_acc = {}
    for label in range(3):
        mask = labels == label
        if mask.sum() > 0:
            per_class_acc[f"class_{label}_accuracy"] = accuracy_score(labels[mask], preds[mask])
    return {"accuracy": accuracy_score(labels, preds), **per_class_acc}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="tensorboard",
    logging_dir=tensorboard_log_dir,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    run_name=RUN_NAME,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[AdvancedTensorBoardCallback(writer)],
)

# --- Train ---
print("Starting training...")
trainer.train(resume_from_checkpoint=checkpoint_to_resume)

# --- Validation Performance Analysis ---
print("Analyzing validation performance...")
val_predictions = trainer.predict(val_dataset)
val_logits = torch.from_numpy(val_predictions.predictions)
val_probs = F.softmax(val_logits, dim=1).numpy()
val_preds = np.argmax(val_probs, axis=1)
val_labels = val_predictions.label_ids

cm = confusion_matrix(val_labels, val_preds)
print("Validation Confusion Matrix:")
print(cm)

# --- Test ---
print("Generating predictions on test set...")
predictions = trainer.predict(test_dataset)
logits = torch.from_numpy(predictions.predictions)
probs = F.softmax(logits, dim=1).numpy()

# --- Submission ---
print("Creating submission file...")
submission_df = pd.DataFrame({"id": df_test["id"]})
submission_df["winner_model_a"] = probs[:, 0]
submission_df["winner_model_b"] = probs[:, 1]
submission_df["winner_model_tie"] = probs[:, 2]

submission_df.to_csv("submission_test.csv", index=False)
print("'submission_test.csv' created.")
writer.close()
