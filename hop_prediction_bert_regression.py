
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import Adam

#config
WEBQSP_TRAIN_PATH = "/kaggle/input/webqsp-anlp/WebQSP.train.json"
WEBQSP_TEST_PATH = "/kaggle/input/webqsp-anlp/WebQSP.test.json"

OUT_DIR = "./hop_model_regression"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 3e-5
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

#loading and preprocessing data
def load_webqsp(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data.get("Questions", []):
        raw_q = item.get("RawQuestion") or item.get("ProcessedQuestion") or ""
        parses = item.get("Parses", [])
        if not parses: continue
        p = parses[0]
        chain = p.get("InferentialChain")
        if chain is None: chain = []
        hop = len(chain)
        if hop == 0: continue
        
        samples.append({
            "question": raw_q.strip(),
            "hop": float(hop), # for regression, Use float hops directly
        })
    return samples

print("Loading WebQSP training data...")
train_val_samples = load_webqsp(WEBQSP_TRAIN_PATH)
print(f"Loaded {len(train_val_samples)} training/validation samples")

print("\nLoading WebQSP test data...")
test_samples = load_webqsp(WEBQSP_TEST_PATH)
print(f"Loaded {len(test_samples)} test samples")

train_val_samples = [s for s in train_val_samples if s["question"]]
test_samples = [s for s in test_samples if s["question"]]
print(f"After filtering empty questions: {len(train_val_samples)} train/val samples, {len(test_samples)} test samples")


unique_hops = sorted(list({int(s["hop"]) for s in train_val_samples}))
print("\nUnique hops found in training data:", unique_hops)
num_labels = 1 

for s in train_val_samples:
    s["label"] = s["hop"]
for s in test_samples:
    s["label"] = s["hop"]

# train/val/test split
train_questions = [s["question"] for s in train_val_samples]
train_labels = [s["label"] for s in train_val_samples]

train_qs, val_qs, train_lbls, val_lbls = train_test_split(
    train_questions, train_labels, test_size=0.1, random_state=SEED
)

test_qs = [s["question"] for s in test_samples]
test_lbls = [s["label"] for s in test_samples]

print("\n--- Data Split ---")
print("Train size:", len(train_qs))
print("Validation size:", len(val_qs))
print("Test size:", len(test_qs))
print("--------------------")

#tokens and dataloaders
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

class HopDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[float], tokenizer: BertTokenizerFast, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(t, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # --- CHANGE FOR REGRESSION: Labels must be floats ---
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item

train_dataset = HopDataset(train_qs, train_lbls, tokenizer, MAX_LEN)
val_dataset = HopDataset(val_qs, val_lbls, tokenizer, MAX_LEN)
test_dataset = HopDataset(test_qs, test_lbls, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#model with label 1 for regression
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

#evaluation function with regression metrics
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> None:
    model.eval()
    raw_preds, rounded_preds, gold = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1) # Squeeze from [batch, 1] to [batch]
            
            raw_preds.extend(logits.cpu().numpy().tolist())
            # --- CHANGE FOR REGRESSION: Round predictions for classification metrics ---
            rounded_preds.extend(torch.round(logits).long().cpu().numpy().tolist())
            gold.extend(labels.long().cpu().numpy().tolist())

    mae = mean_absolute_error(gold, raw_preds)
    acc = accuracy_score(gold, rounded_preds)
    macro_f1 = f1_score(gold, rounded_preds, average="macro", zero_division=0)
    
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    print(f"Accuracy (on rounded): {acc:.4f}")
    print(f"Macro-F1 (on rounded): {macro_f1:.4f}")
    
    target_names = [f"hop={h}" for h in sorted(list(set(gold)))]
    print("\nClassification Report (on rounded predictions):")
    print(classification_report(gold, rounded_preds, target_names=target_names, zero_division=0))
    
    return macro_f1 # Return F1 for best model comparison

# training loop with validation
best_val_f1 = -1.0
print("\nStarting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}")
    
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    print(f"Epoch {epoch} train loss (MSE): {total_loss / len(train_loader):.4f}")
    
    print("\n--- Validation ---")
    val_macro_f1 = evaluate(model, val_loader, DEVICE)
    
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"\nNew best validation F1: {best_val_f1:.4f}. Saving model to {OUT_DIR}")
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)

#test set evaluation
print("\n--- Final Evaluation on Test Set ---")
print("Loading best model from:", OUT_DIR)

best_model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR)
best_model.to(DEVICE)

evaluate(best_model, test_loader, DEVICE)