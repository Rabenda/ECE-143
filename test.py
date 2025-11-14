import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# --- Config ---
MODEL_NAME = "microsoft/deberta-v3-base"

PROTOTYPE_FRAC = 0.05
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 1
TEST_SIZE = 0.1

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "data/sample_submission.csv"
OUTPUT_DIR = "./llm_preference_model_prototype"

# --- Data ---
print("Loading data...")
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sub = pd.read_csv(SUBMISSION_PATH)


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


# --- Model and Tokenizer ---
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# --- Trainer ---
print("Splitting data...")
train_df, val_df = train_test_split(
    df_train, test_size=TEST_SIZE, random_state=42, stratify=df_train["label"]
)

train_dataset = LLMPreferenceDataset(train_df, tokenizer, MAX_LENGTH, augment=True)
val_dataset = LLMPreferenceDataset(val_df, tokenizer, MAX_LENGTH, augment=False)
test_dataset = LLMPreferenceDataset(df_test, tokenizer, MAX_LENGTH, is_test=True)

print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples.")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
    }


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
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- Train ---
print("Starting training...")
trainer.train()

# --- Test ---
print("Training finished. Generating predictions on test set...")
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
preds_indices = np.argmax(logits, axis=1)

# --- Submission ---
print("Creating submission file...")
submission_df = pd.DataFrame({"id": df_test["id"]})
submission_df["winner_model_a"] = (preds_indices == 0).astype(int)
submission_df["winner_model_b"] = (preds_indices == 1).astype(int)
submission_df["winner_model_tie"] = (preds_indices == 2).astype(int)

submission_df.to_csv("submission_test.csv", index=False)
print("'submission_test.csv' created.")
