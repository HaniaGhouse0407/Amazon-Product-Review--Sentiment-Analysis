<div align="center">

# 🎭 Amazon Sentiment Analysis (BERT)

**BERT Fine-Tuned on 500K Amazon Reviews — 92.1% F1 · Word Attribution · Streamlit**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Hania_Ghouse-818CF8?style=flat-square)](https://github.com/HaniaGhouse0407)

</div>

---

## 🧠 Overview

Sentiment classifier fine-tuned on 500K Amazon product reviews. Upgraded from LSTM to BERT for a 14-point F1 improvement. Features word-level attribution highlighting and batch CSV analysis mode.

---

## ✨ Features

- ✅ BERT fine-tuned (bert-base-uncased) — 92.1% F1, 94.3% accuracy
- ✅ 3-class classification: Positive, Negative, Neutral
- ✅ Word-level attribution — highlights sentiment-bearing tokens
- ✅ Batch CSV mode — analyze thousands of reviews at once
- ✅ Confidence probability breakdown per class
- ✅ Compare models: BERT, DistilBERT, RoBERTa

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/HaniaGhouse0407/Amazon-Product-Review--Sentiment-Analysis.git
cd Amazon-Product-Review--Sentiment-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (if needed)
cp .env.example .env
# Edit .env with your API keys

# 4. Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

![transformers](https://img.shields.io/badge/transformers-FFD21E?style=flat-square)  ![torch](https://img.shields.io/badge/torch-EE4C2C?style=flat-square)  ![streamlit](https://img.shields.io/badge/streamlit-FF4B4B?style=flat-square)  ![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square)  ![numpy](https://img.shields.io/badge/numpy-013243?style=flat-square)  ![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square)

---

## 📁 Project Structure

```
Amazon-Product-Review--Sentiment-Analysis/
├── app.py              # Main Streamlit/Gradio application
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## 🎯 Target Roles

> NLP Engineer · ML Engineer · Research Engineer

---

<div align="center">

Made by [Hania Ghouse](https://github.com/HaniaGhouse0407) · 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/hania-ghouse/)
[![Google Scholar](https://img.shields.io/badge/Scholar-Research-4285F4?style=flat-square&logo=google-scholar)](https://scholar.google.com/citations?user=iVWuM4wAAAAJ)

</div>
