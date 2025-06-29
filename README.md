# 🧠 EMOLEX: Emotion and Language Exploration for Mental Health

A mental health sentiment classification project using NLP to detect psychological states from short text. This project explores multiple models — from classical baselines to LSTMs and transformers (like BERT) — to identify sentiments such as **Anxiety**, **Depression**, **Suicidal Ideation**, and others from user-generated text.

---

## 📌 Project Description

The goal of EMOLEX is to develop a multi-class classifier that predicts the mental health condition underlying a given text statement. The project benchmarks:

- **Transformer models** (BERT)
- **Bidirectional LSTM**
- **Classical ML models** (TF-IDF + Logistic Regression)

We aim to better understand the performance trade-offs between accuracy and efficiency across model types while enabling mental health research through NLP.

---

## 📁 Repository Structure

```bash
NLP_Project/
├── LICENSE               # MIT License
├── README.md             # Repository overview with setup instructions
├── archive               # Old Stuff
├── data                  # Datasets (or download scripts)
├── documents             # Documentation, architecture, research notes
├── notebooks             # Development and experiment notebooks
└── requirements.txt      # Project dependencies
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/NLP_Project.git
cd NLP_Project
```

### 2. Create a Virtual Environment
```bash
conda create -n emolex python=3.10
conda activate emolex
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Place your dataset in the `data/` folder. You can use any CSV with `text` and `label` columns. For example:
```csv
text,label
"I can't sleep and feel exhausted.",Anxiety
"Everything feels meaningless.",Depression
```

### 5. Run Experiments
Use the notebooks in `/notebooks` to explore preprocessing, model training, and evaluation.

---

## 🧪 Models & Evaluation
- ✅ BERT fine-tuning
- ✅ Bidirectional LSTM
- ✅ Logistic Regression with TF-IDF
- 🔎 Evaluation: Accuracy, Macro F1, Precision/Recall, Confusion Matrix

---

## 📊 Benchmarks
Model | Accuracy | F1 Score | Inference Time
------|----------|----------|----------------
BERT  |  XX%     | XX%      | High
LSTM  |  XX%     | XX%      | Medium
TF-IDF|  XX%     | XX%      | Low

_(Replace with your actual results)_

---

## 📄 License
MIT License — feel free to use, share, and modify.

---

## 🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 🧠 Project Maintainers
- [Jiajin Zhou](https://github.com/jiajinz)
- [Jie Lian](https://github.com/jiajinz) _(Replace with Jie's github username)_
- [Peter Mink](https://github.com/jiajinz) _(Replace with Peter's github username)_
- [Curtis Neiderer](https://github.com/cneiderer)
- Contributors welcome!