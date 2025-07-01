# Milestone 2: Model Development

## 1. Research and Methods

### <u>Objectives</u>
This project explores how modern NLP techniques can support mental health screening and analysis with the aim to build a robust, fine-tuned NLP model capable of detecting and classifying mental health-related sentiments expressed in short text (e.g., social media posts, journal entries, etc.). The model will predict one of several mental health categories — including Anxiety, Depression, Suicidal Ideation, Stress, Bipolar Disorder, Personality Disorder, and Normal — based on user-generated text.

### <u>Literature Review</u>
_(Ths section needs to be populated)_

### <u>Benchmarking</u>
* __Classical Baseline Model__: A traditional machine learning model (e.g., LR, SVM, etc.) trained on TF-IDF features, offering a lightweight benchmark for comparison.
* __LSTM-Based Model__: A deep learning baseline using LSTM-based architecture serving as a middle ground between classical and transformer-based methods.
* __Transformer-Based Model__: A transformer (e.g., BERT, DistilBERT, etc.) tuned to classify user-generated text into one of the seven mental health categories.
* __Model Evaluation & Comparison__:
  * Quantitative metrics: Accuracy, Macro F1-score, Precision, Recall, and Confusion Matrix
  * Efficiency metrics: Training time and resource usage

__Comparative Analysis__ of the performance across all three model types:

Model        | Accuracy | Precision | Recall | F1 Score | Training Time
-------------|----------|-----------|--------|----------|---------------
DistilBERT   |  0.80    | 0.73      | 0.80   | 0.75     | Very High (~3K s/epoch)
BERT         |  0.81    | 0.78      | 0.78   | 0.78     | High (~1800 s/epoch)
BiLSTM       |  0.77    | 0.72      | 0.69   | 0.70     | Low (~40 s/epoch)
LSTM         |  0.75    | 0.70      | 0.67   | 0.68     | Medium (~240 s/epoch)
TF-IDF + LR  |  0.76    | 0.74      | 0.67   | 0.70     | Very Low, Negligible (~21s total)
TF-IDF + SVM |  0.75    | 0.73      | 0.70   | 0.71     | Very Low, Negligible (~16s total)

### <u>Preliminary Experiments</u>
_(TBD)_

## 2. Model Implementation


### <u>Framework Selection</u>


### <u>Dataset Preparation</u>


### <u>Model Development</u>


### <u>Training & Fine-Tuning</u>


### <u>Evaluation & Metrics</u>


## 3. GitHub Repository Setup & Code Management
The project repository is located here:
https://github.com/jiajinz/NLP_Project/tree/curtis


### Repository Structure
```
NLP_Project/
├── LICENSE               # MIT License
├── README.md             # Repository overview with setup instructions
├── archive               # Old Stuff
├── data                  # Datasets (or download scripts)
├── documents             # Documentation, architecture, research notes
├── notebooks             # Development and experiment notebooks
└── requirements.txt      # Project dependencies
```
Note: To get the most up-to-date repository structure, within a `bash shell` run `tree .` within the top-level of the repository.

### Version Control & Collaboration
Our team used Git as the primary version control system to manage code development and collaboration throughout the project. 

We adopted a branching strategy where each team member worked on their own individual branches (e.g., `jie`, `jiajin`, `peter`, and `curtis`) and merged into the shared `main` branch via pull requests.

To coordinate work we used Slack to track tasks and progress. This setup enabled asynchronous collaboration, efficient conflict resolution, and a clear history of project evolution. 

### Documentation
The README file with setup instructions is located here:
https://github.com/jiajinz/NLP_Project/blob/curtis/README.md

### Submission & Accessibility
The project is hosted in a public repository on GitHub, located at:
https://github.com/jiajinz/NLP_Project/tree/curtis