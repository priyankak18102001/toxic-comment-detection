import pandas as pd
import numpy as np
import streamlit as st
import pickle
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------
# Settings
# --------------------------
st.set_page_config(page_title = "Toxic Comment Detector" , page_icon= "🛡️", layout="wide")

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_len = 150   

metrics_path = os.path.join(os.path.dirname(__file__), "metrics.json")

# --------------------------
# Load Model + Tokenizer
# --------------------------

@st.cache_resource
def load_artifatcs():
    model = load_model("toxicity_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifatcs()

# --------------------------
# Helper function
# --------------------------

def predict_comment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq , maxlen = max_len, padding = "post" , truncating = "post")
    pred = model.predict(padded)[0]
    return pred 

# --------------------------
# Title
# --------------------------

st.title("🛡️ Toxic Comment Detection App")
st.write("Enter a comment and get toxicity prediction in real time")

# --------------------------
# Sidebar
# --------------------------

st.sidebar.header("Options")
show_metrics = st.sidebar.checkbox("Show Model Performance Metrics" , value = True)
show_sample = st.sidebar.checkbox("Show Sample Test Cases", value = True)
show_insights = st.sidebar.checkbox("Show Dataset Insights" , value = True)

# --------------------------
# Layout
# --------------------------

col1 , col2 = st.columns([1.2,1])

# --------------------------
# Real-time prediction
# --------------------------

with col1:
    st.subheader("Real-time Toxicity Prediction")
    user_comment = st.text_area("Type your comment here:", height=150)

    if st.button("predict"):
        if user_comment.strip() == "":
            st.warning("please enter a comment first")
        else:
            pred = predict_comment(user_comment)

            st.success("prediction Done")
            result_df = pd.DataFrame({
                "Label" : target_cols,
                "probability" : pred
            }).sort_values(by = "probability", ascending= False) 

            st.dataframe(result_df , use_container_width=True)
            st.subheader(" Toxicity Score Chart") 
            st.bar_chart(result_df.set_index("Label")["probability"])



# --------------------------
# CSV Bulk prediction
# ------------------------

with col2:
    st.subheader("Bulk Prediction using CSV Upload")

    st.info("Upload a CSV with one column named: **comment_text**")

    uploded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploded_file is not None:
        df = pd.read_csv(uploded_file)

        if "comment_text" not in df.columns:
            st.error(" CSV must contain a column named 'comment_text'")
        else:
            st.write(" Uploaded Data Preview")
            st.dataframe(df.head(),use_container_width=True)

            if st.button("Run Bulk Prediction"):
                preds = []
                for text in df["comment_text"].astype(str):
                    pred = predict_comment(text)
                    preds.append(pred)

                preds = np.array(preds)

                out_df = df.copy()
                for i , col in enumerate(target_cols):
                    out_df[col] = preds[:,i]


                st.success(" Bulk predictions completed!")
                st.dataframe(out_df.head(), use_container_width=True )

                # Download results
                csv_data = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇ Download Prediction Results",
                    data=csv_data,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )   

# --------------------------
# Model Performance Metrics 
# ------------------------

if show_metrics:
    st.markdown("---")
    st.subheader("📌 Model Performance Metrics")

    st.write("✅ Current Working Directory:", os.getcwd())
    st.write("✅ Files in folder:", os.listdir())

    try:
        st.write(" Trying to open metrics.json ...")

        with open("metrics.json", "r") as f:
            metrics = json.load(f)

        st.success(" metrics.json loaded successfully!")

        if metrics is None:
            st.error(" metrics.json loaded as None. File may be empty or contains null.")
        else:
            st.write("keys inside metrics.json:", list(metrics.keys()))

        st.write(" Macro Avg F1-score:", metrics["macro avg"]["f1-score"])
        st.write(" Weighted Avg F1-score:", metrics["weighted avg"]["f1-score"])

        rows = []
        for label in target_cols:
            rows.append({
                "Label": label,
                "Precision": metrics[label]["precision"],
                "Recall": metrics[label]["recall"],
                "F1-score": metrics[label]["f1-score"]
            })

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

    except Exception as e:
        st.error(" Error while loading metrics.json")
        st.exception(e)
# --------------------------
# Sample test cases
# --------------------------

if show_sample:
    st.markdown("---")
    st.subheader("🧪 Sample Test Cases")

    samples = [
        "I love your work! Amazing job ",
        "You are stupid and useless",
        "I will kill you",
        "This is the worst thing ever",
        "Go away idiot"
    ]   

    for s in samples:
        st.write("🔹", s)     

# --------------------------
# Dataset Insights (optional)
# -------------------------- 

if show_insights:
    st.markdown("---")
    st.subheader(" Dataset Insights (Optional)")

    try:
        train_df = pd.read_csv("train.csv")
        st.write("Dataset Shape:", train_df.shape)
        st.write("Available Label Columns:", target_cols)

        label_counts = train_df[target_cols].sum().sort_values(ascending=False)
        st.bar_chart(label_counts)

    except:
        st.warning("Dataset file not found in `data/train.csv`. Upload it if you want dataset insights.")    


                             








































































