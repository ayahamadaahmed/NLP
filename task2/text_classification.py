# text_classification.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = r"./results_toxic"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def classify_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return "Toxic" if pred == 1 else "Non-Toxic"

if __name__ == "__main__":
    print(classify_text("You're the worst person ever!"))   # توقع: Toxic
    print(classify_text("I really love this project."))     # توقع: Non-Toxic)
