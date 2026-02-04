# 🛡️ Toxic Comment Detection using Deep Learning + Streamlit

This project is a **multi-label toxic comment classification system** that detects different types of toxicity in user comments using a **Deep Learning (LSTM) model**.  
It is deployed as an interactive **Streamlit web application** that supports both **real-time prediction** and **bulk CSV prediction**.

---

##  Features
 Real-time toxicity prediction for user input comments  
Predicts multiple toxicity categories:
- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

Bulk prediction using CSV file upload  
 Download predictions as a CSV file  
 Shows dataset insights and model metrics in the dashboard

---

## 🧠 Model Used
- Tokenization + Padding
- Embedding Layer
- BiLSTM (Bidirectional LSTM)
- Sigmoid output layer for **multi-label classification**

Loss Function: **Binary Cross Entropy**

---

##  Project Structure
```bash
Toxicity Detection project/
│── app.py
│── requirements.txt
│── README.md
│── toxicity_model.h5
│── tokenizer.pkl
│── metrics.json
│── train.csv
│── test.csv

Installation & Setup
1️⃣ Create and Activate Virtual Environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

Install Dependencies
pip install -r requirements.txt

Run Streamlit App
streamlit run app.py

Bulk CSV Prediction Format

Your CSV must contain a column named:

comment_text

Example:

comment_text
I love your work!
You are stupid and useless

Model Performance Metrics

The app loads model evaluation metrics from:

 metrics.json

This file is generated after training using classification_report().

Dataset Used

Jigsaw Toxic Comment Classification Dataset (multi-label toxic comments).

Author

Priyanka Kumawat
LinkedIn: www.linkedin.com/in/priyanka-kumawat-7177092a3
