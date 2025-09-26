# train_toxic.py
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# تحميل CSV
df = pd.read_csv(r"C:\Users\ayaha\Downloads\archive(2)\train.csv")

# عمود labels (binary)
df["labels"] = (df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].sum(axis=1) > 0).astype(int)

# نختار الأعمدة المهمة
df = df[["comment_text", "labels"]].rename(columns={"comment_text": "text"})

# تحويله إلى HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# split train/test
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# تجهيز tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset = dataset.map(preprocess, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# تحميل الموديل
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# إعدادات التدريب
training_args = TrainingArguments(
    output_dir="./results_toxic",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-4,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(5000)),  # subset للتجربة
    eval_dataset=dataset["test"].shuffle(seed=42).select(range(1000)),
    tokenizer=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./results_toxic")
    tokenizer.save_pretrained("./results_toxic")

    # اختبار سريع
    text = "You're the worst person ever!"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits).item()
    print("Logits:", logits)
    print("Predicted class:", "Toxic" if pred == 1 else "Non-Toxic")
