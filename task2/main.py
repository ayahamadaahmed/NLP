# main.py
import streamlit as st
from imagecaption import generate_caption
from text_classification import classify_text, model, tokenizer
from database import save_to_csv, load_database
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from datasets import Dataset
import torch
import base64

st.set_page_config(page_title="🛡️ Toxic Content Classifier", page_icon="🛡️", layout="centered")
st.title("🛡️ Toxic Content Classifier")

option = st.radio("Choose input type:", ["Text", "Image"])

# 📝 نص
if option == "Text":
    user_input = st.text_area("✍️ Enter your text:")
    if st.button("Classify Text"):
        if user_input.strip():
            label = classify_text(user_input)
            st.success(f"Result: **{label}**")
            save_to_csv(user_input, label)
        else:
            st.warning("⚠️ Please enter some text.")

# 🖼️ صورة
elif option == "Image":
    image_file = st.file_uploader("📷 Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        caption = generate_caption(image_file)
        st.image(image_file, caption="Uploaded Image", use_column_width=True)
        st.info(f"Generated Caption: **{caption}**")
        if st.button("Classify Caption"):
            label = classify_text(caption)
            st.success(f"Result: **{label}**")
            save_to_csv(caption, label)

# 📑 قاعدة البيانات
if st.button("📑 View Database"):
    db = load_database()
    if not db.empty:
        st.dataframe(db, use_container_width=True)
    else:
        st.info("Database is empty.")

# ⬇️ زر التحميل
if st.button("⬇️ Download Database"):
    db = load_database()
    if not db.empty:
        csv = db.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="database.csv">Download database.csv</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("Database is empty, nothing to download.")

# 📊 تقييم الموديل
def evaluate_model(test_texts, test_labels, model, tokenizer):
    preds, labels = [], []
    for text, label in zip(test_texts, test_labels):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits).item()
        preds.append(pred)
        labels.append(label)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1

if st.button("📊 Evaluate Model"):
    st.info("Evaluating model on Kaggle toxic test set...")
    
    # تحميل جزء من test split (لو عايز تقلل خليه 200 بدل 1000)
    df = pd.read_csv(r"C:\Users\ayaha\Downloads\archive(2)\train.csv")
    df["labels"] = (df[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].sum(axis=1) > 0).astype(int)

    test_df = df.sample(500, random_state=42)  # subset
    texts = test_df["comment_text"].tolist()
    labels = test_df["labels"].tolist()

    acc, f1 = evaluate_model(texts, labels, model, tokenizer)

    st.success("✅ Evaluation Complete!")
    st.metric("Accuracy", f"{acc:.2f}")
    st.metric("F1 Score", f"{f1:.2f}")
    st.write(f"📊 Accuracy: {acc:.4f}, F1: {f1:.4f}")
