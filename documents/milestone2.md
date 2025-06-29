# Milestone 2: Model Development

## 1. Research and Methods

### <u>Objectives</u>
This project explores how modern NLP techniques can support mental health screening and analysis with the aim to build a robust, fine-tuned NLP model capable of detecting and classifying mental health-related sentiments expressed in short text (e.g., social media posts, journal entries, etc.). The model will predict one of several mental health categories — including Anxiety, Depression, Suicidal Ideation, Stress, Bipolar Disorder, Personality Disorder, and Normal — based on user-generated text.

### <u>Literature Review</u>
_(TBD, Needs to be populated)_

### <u>Benchmarking</u>
* __Classical Baseline Model__: A traditional machine learning model (e.g., logistic regression or SVM) trained on TF-IDF features, offering a lightweight benchmark for comparison.
* __LSTM-Based Model__: A deep learning baseline using a Bidirectional LSTM serving as a middle ground between classical and transformer-based methods.
* __Transformer-Based Model__: A fine-tuned transformer (e.g., BERT) trained to classify user-generated text into one of seven mental health categories.
* __Model Evaluation & Comparison__:
  * Quantitative metrics: Accuracy, Macro F1-score, Precision, Recall, and Confusion Matrix
  * Efficiency metrics: Training time and resource usage

__Comparative Analysis__ of the performance across all three model types:

_(TBD, Needs comparison table to be added)_


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
├── LICENSE
├── README.md
├── archive
│   └── ie7500_group2_initialProjectProposal--SkyComm-AIDE .docx
├── data
│   └── mental_health_sentiment.csv
├── documents
│   └── ie7500_group2_updatedProjectProposal.docx
├── notebooks
│   ├── DistilBERT_sentiment_pipeline_cleanran.ipynb
│   └── initial_project_notebook.ipynb
└── requirements.txt
```
Note: To get the most up-to-date repository structure, within a `bash shell` run `tree .` within the top-level of the repository.

### Version Control & Collaboration
Our team used Git as the primary version control system to manage code development and collaboration throughout the project. We hosted the project on GitHub, maintaining a clean and modular repository structure.

We adopted a branching strategy where each team member worked on their own individual branches (e.g., `jie`, `jiajin`, `peter`, and `curtis`) and merged into the shared `main` branch via pull requests.

To coordinate work we used Slack to track tasks and progress. This setup enabled asynchronous collaboration, efficient conflict resolution, and a clear history of project evolution. 

### Documentation
The README file with setup instructions is located here:
https://github.com/jiajinz/NLP_Project/blob/curtis/README.md