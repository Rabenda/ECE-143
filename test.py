import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

PROTOTYPE_FRAC = 1
MAX_LENGTH = 512
BATCH_SIZE = 30
LEARNING_RATE = 2e-5
EPOCHS = 1
TEST_SIZE = 0.1

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "data/sample_submission.csv"
OUTPUT_DIR = "./llm_preference_model_prototype"
TENSORBOARD_DIR = "./tf-logs"

# --- Data ---
print("Loading data...")
df_train = pd.read_csv(TRAIN_PATH, engine="python")
df_test = pd.read_csv(TEST_PATH, engine="python")
df_sub = pd.read_csv(SUBMISSION_PATH, engine="python")


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
    print("Class distribution in prototype set:")
    print(df_train["label"].value_counts(normalize=True))


# --- PyTorch Dataset ---
class LLMPreferenceDataset(Dataset):
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

    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        resp_a = str(self.response_as[idx])
        resp_b = str(self.response_bs[idx])

        if not self.is_test:
            label = self.labels[idx]

        if self.augment and torch.rand(1).item() > 0.5:
            resp_a, resp_b = resp_b, resp_a
            if not self.is_test:
                if label == 0:
                    label = 1
                elif label == 1:
                    label = 0

        text = prompt + self.tokenizer.sep_token + resp_a + self.tokenizer.sep_token + resp_b

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        if self.is_test:
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(label, dtype=torch.long),
            }


# --- TensorBoard Callback ---
class TensorBoardCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training"""
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if metrics is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"eval/{key}", value, state.global_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log gradients and weights periodically"""
        if state.global_step % 50 == 0 and model is not None:
            # Log gradient norms
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    # Log individual layer gradient norms
                    self.writer.add_scalar(f"gradients/{name}", param_norm, state.global_step)

            total_norm = total_norm**0.5
            self.writer.add_scalar("gradients/total_norm", total_norm, state.global_step)

            # Log weight distributions for key layers
            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    self.writer.add_histogram(f"weights/{name}", param.data, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Close writer at end of training"""
        self.writer.close()


# --- Model and Tokenizer ---
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# --- Initialize TensorBoard Writer ---
print(f"Initializing TensorBoard. Run: tensorboard --logdir={TENSORBOARD_DIR}")
writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

# Log hyperparameters
hparams = {
    "model": MODEL_NAME,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "prototype_frac": PROTOTYPE_FRAC,
}
writer.add_text("hyperparameters", str(hparams), 0)

# --- Trainer ---
print("Splitting data...")
train_df, val_df = train_test_split(
    df_train, test_size=TEST_SIZE, random_state=42, stratify=df_train["label"]
)

train_dataset = LLMPreferenceDataset(train_df, tokenizer, MAX_LENGTH, augment=True)
val_dataset = LLMPreferenceDataset(val_df, tokenizer, MAX_LENGTH, augment=False)
test_dataset = LLMPreferenceDataset(df_test, tokenizer, MAX_LENGTH, is_test=True)

print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")

# Log dataset statistics
writer.add_scalar("data/train_size", len(train_dataset), 0)
writer.add_scalar("data/val_size", len(val_dataset), 0)
writer.add_scalar("data/test_size", len(test_dataset), 0)

# Log class distribution
class_dist = train_df["label"].value_counts(normalize=True).sort_index()
for label, freq in class_dist.items():
    writer.add_scalar(f"data/class_{int(label)}_freq", freq, 0)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    # Calculate per-class accuracy
    per_class_acc = {}
    for label in range(3):
        mask = labels == label
        if mask.sum() > 0:
            per_class_acc[f"class_{label}_accuracy"] = accuracy_score(labels[mask], preds[mask])

    metrics = {"accuracy": accuracy_score(labels, preds), **per_class_acc}

    return metrics


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="tensorboard",
    logging_dir=TENSORBOARD_DIR,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TensorBoardCallback(writer)],
)

# --- Train ---
print("Starting training...")
trainer.train()

# --- Validation Performance Analysis ---
print("Analyzing validation performance...")
val_predictions = trainer.predict(val_dataset)
val_logits = torch.from_numpy(val_predictions.predictions)
val_probs = F.softmax(val_logits, dim=1).numpy()
val_preds = np.argmax(val_probs, axis=1)
val_labels = val_predictions.label_ids

# Log confusion matrix data
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_labels, val_preds)
print("Validation Confusion Matrix:")
print(cm)

# Log final validation metrics
final_val_acc = accuracy_score(val_labels, val_preds)
writer.add_scalar("final/validation_accuracy", final_val_acc, 0)

# Log per-class performance
for i in range(3):
    class_mask = val_labels == i
    if class_mask.sum() > 0:
        class_acc = accuracy_score(val_labels[class_mask], val_preds[class_mask])
        writer.add_scalar(f"final/class_{i}_accuracy", class_acc, 0)
        print(f"Class {i} accuracy: {class_acc:.4f}")

# --- Test ---
print("Generating predictions on test set...")
predictions = trainer.predict(test_dataset)
logits = torch.from_numpy(predictions.predictions)
probs = F.softmax(logits, dim=1).numpy()

# Log prediction distribution
pred_classes = np.argmax(probs, axis=1)
for i in range(3):
    class_ratio = (pred_classes == i).sum() / len(pred_classes)
    writer.add_scalar(f"test/predicted_class_{i}_ratio", class_ratio, 0)
    print(f"Test set - Class {i} predicted ratio: {class_ratio:.4f}")

# Log prediction confidence
avg_confidence = np.max(probs, axis=1).mean()
writer.add_scalar("test/average_confidence", avg_confidence, 0)
writer.add_histogram("test/prediction_confidence", np.max(probs, axis=1), 0)

# --- Submission ---
print("Creating submission file...")
submission_df = pd.DataFrame({"id": df_test["id"]})
submission_df["winner_model_a"] = probs[:, 0]
submission_df["winner_model_b"] = probs[:, 1]
submission_df["winner_model_tie"] = probs[:, 2]

submission_df.to_csv("submission_test.csv", index=False)
print("'submission_test.csv' created.")

# Close TensorBoard writer
writer.close()
print(f"\nTensorBoard logs saved to: {TENSORBOARD_DIR}")
print(f"To view, run: tensorboard --logdir={TENSORBOARD_DIR}")
