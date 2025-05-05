import argparse
import json
import torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

class BugDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples  = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        bug_text, code_snip, label = self.examples[idx]
        enc = self.tokenizer(
            bug_text,
            code_snip,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item, torch.tensor(label)

def load_data(path, bug_key, code_key):
    examples = []
    first = True
    with open(path, 'r', encoding='utf-8') as f:
        for row in f:
            obj = json.loads(row)
            if first:
                print("Detected JSON keys:", list(obj.keys()))
                first = False
            bug_text  = obj.get(bug_key)
            code_snip = obj.get(code_key)
            label     = obj.get('label')
            if bug_text is None or code_snip is None or label is None:
                continue
            examples.append((bug_text, code_snip, int(label)))
    return examples

def main(args):
    # 1️⃣ Load & split
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    data      = load_data(args.data_path, bug_key=args.bug_key, code_key=args.code_key)
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

    # 2️⃣ DataLoaders
    train_ds     = BugDataset(train_data, tokenizer)
    val_ds       = BugDataset(val_data,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    # 3️⃣ Model & optimizer
    model     = AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/codebert-base", num_labels=2
                )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4️⃣ Training loop with progress bar
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        # tqdm progress bar over batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", ncols=80)
        for batch_inputs, labels in pbar:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            labels       = labels.to(device)

            outputs = model(**batch_inputs, labels=labels)
            loss    = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"\n→ Epoch {epoch}/{args.epochs} — avg train loss: {avg_train_loss:.4f}")

        # 5️⃣ Validation (optional)
        if args.eval_data_path:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs = {k: v.to(device) for k, v in val_inputs.items()}
                    val_labels = val_labels.to(device)
                    logits     = model(**val_inputs).logits
                    preds      = torch.argmax(logits, dim=1)
                    correct   += (preds == val_labels).sum().item()
                    total     += val_labels.size(0)
            acc = correct / total if total > 0 else 0
            print(f"  ↳ Validation accuracy: {acc:.4f}\n")
            model.train()

    elapsed = time.time() - start_time
    print(f"🏁 Training finished in {elapsed/60:.2f} minutes")

    # 6️⃣ Save weights
    torch.save(model.state_dict(), args.save_path)
    print(f"✅ Saved fine-tuned model to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for bug localization")
    parser.add_argument("--data_path",      required=True, help="Path to JSONL training data")
    parser.add_argument("--eval_data_path", default=None, help="Path to JSONL validation data (optional)")
    parser.add_argument("--save_path",      default="models/model.pt", help="Where to save the model")
    parser.add_argument("--bug_key",        default="bug",   help="JSON key for bug-report text")
    parser.add_argument("--code_key",       default="func",  help="JSON key for code snippet")
    parser.add_argument("--epochs",     type=int,   default=3,    help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    main(args)
