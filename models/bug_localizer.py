import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BugLocalizer:
    def __init__(self, model_path="models/model.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/codebert-base", num_labels=2
        )
        try:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print("✅ Loaded fine-tuned model.")
        except Exception:
            print("⚠️  No fine-tuned weights found; using base model.")
        self.model.eval()

    def predict(self, code: str, bug_report: str):
        lines = code.splitlines() or [""]
        scores = []
        for ln in lines:
            enc = self.tokenizer(
                bug_report, ln,
                truncation=True, padding="max_length", max_length=256,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**enc).logits
                score  = torch.softmax(logits, dim=1)[0,1].item()
            scores.append(score)
        return list(zip(range(1, len(lines)+1), scores))
