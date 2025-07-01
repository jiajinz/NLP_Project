# <img src="./emolex_logo.png" align="left" alt="Sample Image" class="image-left" width="80px" height="80px" style="padding: 10px"/> EMOLEX: Emotion and Language Exploration for Mental Health<br>


A mental health sentiment classification project using NLP to detect psychological states from short text. This project explores multiple models — from classical baselines to LSTMs and transformers (e.g., BERT) — to identify sentiments such as Anxiety, Depression, Suicidal Ideation, Stress, Bipolar Disorder, Personality Disorder, and Normal from user-generated text.

## 📌 Project Description

The goal of EMOLEX is to develop a multi-class classifier that predicts the mental health condition underlying a given text statement. The project benchmarks:

- **Transformer models** (BERT)
- **Bidirectional LSTM**
- **Classical ML models** (TF-IDF + Logistic Regression)

We aim to better understand the performance trade-offs between accuracy and efficiency across model types while enabling mental health research through NLP.

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
_(This section needs to updated development progresses.)_

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

_(This section needs to updated and flushed out.)_

## 🧪 Models & Evaluation
- ✅ BERT fine-tuning
- ✅ Bidirectional LSTM
- ✅ Classical: Logistic Regression with TF-IDF
- 🔎 Evaluation: Accuracy, Macro F1, Precision/Recall, Confusion Matrix

## 📊 Benchmarks
Model | Accuracy | Precision | Recall | F1 Score | Inference Time
------|----------|-----------|--------|----------|---------------
BERT  |  0.80    | 0.73      | 0.80   | 0.75     | High (~3K seconds per epoch)
LSTM  |  0.75    | 0.70      | 0.69   | 0.70     | Medium (~30 seconds per epoch)
Classical |  X.XX    | X.XX      | X.XX   | X.XX     | Low (~XX seconds per epoch)

_(This section needs to updated as results come in.)_

## 📄 License
MIT License — feel free to use, share, and modify.

## 🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 🧠 Project Maintainers
- [Jiajin Zhou](mailto:zhou.j@northeastern.edu)
- [Jie Lian](mailto:lian.j@northeastern.edu)
- [Peter Mink](mailto:mink.p@northeastern.edu)
- [Curtis Neiderer](mailto:neiderer.c@northeastern.edu)
- Contributors welcome!
