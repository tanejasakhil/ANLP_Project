
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import Adam

WEBQSP_TRAIN_PATH = "/kaggle/input/webqsp-anlp/WebQSP.train.json"  # <--- CHANGE TO YOUR TRAIN FILE PATH
WEBQSP_TEST_PATH = "/kaggle/input/webqsp-anlp/WebQSP.test.json"   # <--- CHANGE TO YOUR TEST FILE PATH

OUT_DIR = "./hop_model"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 3e-5
WEIGHT_DECAY = 0.01
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

#loading and preprocessing data
def load_webqsp(path: str) -> List[Dict]:
    """Load WebQSP JSON file and return list of cleaned samples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    
    for item in data.get("Questions", []):
        raw_q = item.get("RawQuestion") or item.get("ProcessedQuestion") or ""
        parses = item.get("Parses", [])
        if not parses:
            continue
            
        p = parses[0]
        chain = p.get("InferentialChain")
        
        if chain is None:
            chain = []
            
        hop = len(chain)
        
        if hop == 0:
            continue
            
        answers = [a.get("EntityName") for a in p.get("Answers", []) if a.get("EntityName")]
        topic_mid = p.get("TopicEntityMid")
        samples.append({
            "question": raw_q.strip(),
            "hop": int(hop),
            "relations": chain,
            "answers": answers,
            "topic_mid": topic_mid
        })
    return samples


print("Loading WebQSP training data...")
train_val_samples = load_webqsp(WEBQSP_TRAIN_PATH)
print(f"Loaded {len(train_val_samples)} training/validation samples")

print("\nLoading WebQSP test data...")
test_samples = load_webqsp(WEBQSP_TEST_PATH)
print(f"Loaded {len(test_samples)} test samples")

# Filter out empty questions
train_val_samples = [s for s in train_val_samples if s["question"]]
test_samples = [s for s in test_samples if s["question"]]
print(f"After filtering empty questions: {len(train_val_samples)} train/val samples, {len(test_samples)} test samples")

#Create mapping from all found hops to zero-indexed labels
# Build label set from training data only
unique_hops = sorted(list({s["hop"] for s in train_val_samples}))
print("\nUnique hops found in training data (all will be used):", unique_hops)

# hop_to_label: {1: 0, 2: 1, 3: 2, ...}
# label_to_hop: {0: 1, 1: 2, 2: 3, ...}
hop_to_label = {h: i for i, h in enumerate(unique_hops)}
label_to_hop = {v: k for k, v in hop_to_label.items()}
num_labels = len(unique_hops)
print(f"Created {num_labels} classes. Mapping labels back to hops for reporting.")

# Apply zero-indexed labels to both datasets
for s in train_val_samples:
    s["label"] = hop_to_label[s["hop"]]
for s in test_samples:
    # Test samples with hops not seen in the training data will be ignored
    s["label"] = hop_to_label.get(s["hop"], -1) 

# Filter out test samples with labels not in training data
original_test_count = len(test_samples)
test_samples = [s for s in test_samples if s["label"] != -1]
if len(test_samples) != original_test_count:
    print(f"Filtered out {original_test_count - len(test_samples)} test samples with hops not present in the training set.")

#train val test split
train_questions = [s["question"] for s in train_val_samples]
train_labels = [s["label"] for s in train_val_samples]

train_qs, val_qs, train_lbls, val_lbls = train_test_split(
    train_questions, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
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
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizerFast, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

train_dataset = HopDataset(train_qs, train_lbls, tokenizer, MAX_LEN)
val_dataset = HopDataset(val_qs, val_lbls, tokenizer, MAX_LEN)
test_dataset = HopDataset(test_qs, test_lbls, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#model, optimizer, scheduler
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

optimizer = Adam(model.parameters(), lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.06 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

#evaluation function with classification metrics
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float, List, List]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            
            preds.extend(batch_preds.cpu().numpy().tolist())
            gold.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", zero_division=0)
    
    return acc, macro_f1, gold, preds

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

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")

    val_acc, val_macro_f1, val_gold, val_preds = evaluate(model, val_loader, DEVICE)
    print(f"Validation Acc: {val_acc:.4f}  Macro-F1: {val_macro_f1:.4f}")
    
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"New best validation F1: {best_val_f1:.4f}. Saving model to {OUT_DIR}")
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)

print("\nTraining finished. Best validation macro-F1:", best_val_f1)

#test set evaluation
print("\n--- Final Evaluation on Test Set ---")
print("Loading best model from:", OUT_DIR)

best_model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR)
best_model.to(DEVICE)

test_acc, test_macro_f1, test_gold, test_preds = evaluate(best_model, test_loader, DEVICE)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Final Test Macro-F1: {test_macro_f1:.4f}")

#Use the label_to_hop mapping to show results in terms of original hop numbers
target_names = [f"hop={label_to_hop[i]}" for i in sorted(label_to_hop.keys())]
print("\nClassification Report (Test Set):")
print(classification_report(test_gold, test_preds, target_names=target_names, zero_division=0))