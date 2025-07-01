# Milestone 2: Model Development

## 1. Research and Methods

### <u>Objectives</u>
This project explores how modern NLP techniques can support mental health screening and analysis with the aim to build a robust, fine-tuned NLP model capable of detecting and classifying mental health-related sentiments expressed in short text (e.g., social media posts, journal entries, etc.). The model will predict one of several mental health categories — including Anxiety, Depression, Suicidal Ideation, Stress, Bipolar Disorder, Personality Disorder, and Normal — based on user-generated text.

### <u>Research and Literature Review</u>

Finding an appropriate dataset was a critical step in ensuring the success and relevance of our sentiment classification project. Our goal was to identify a dataset that contained diverse and representative text samples with labels. We evaluated several publicly available datasets from academic sources, Kaggle, and otherwise free-and-open resources, prioritizing those that offered clearly defined labels and linguistic diversity. The selected dataset, [_Sentiment Analysis for Mental Health_. Kaggle. (Retrieved 2025)](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health), consisted of short, conversational-style text entries, each annotated with one of seven sentiment categories relevant to mental health. We verified that the dataset was balanced enough for meaningful classification while still reflecting real-world challenges such as label imbalance and ambiguous phrasing. Additionally, we ensured that the dataset adhered to privacy standards and did not include any personally identifiable information. This careful selection process ensured that the data was both ethically sound and well-suited to train models capable of supporting nuanced mental health analysis.

As part of the project’s foundation, we conducted a thorough literature review to understand recent advances in sentiment analysis and to identify the most effective modeling approaches. We examined peer-reviewed papers, conference proceedings (ACL, EMNLP, NeurIPS), and arXiv preprints that focus on text classification. Notably, we explored studies that applied transformer models such as BERT and DistilBERT, which have consistently outperformed traditional methods in various NLP tasks due to their ability to capture deep contextual semantics. For instance, recent work using BERT fine-tuned on Reddit mental health posts demonstrated strong performance in identifying signs of depression and suicidal ideation. We also reviewed research highlighting the effectiveness of LSTM models for handling sequential text data and capturing emotional patterns over time. In addition, we analyzed papers that benchmark classical models like Logistic Regression and SVMs with engineered features (e.g., TF-IDF) as competitive baselines. From this review, we identified best practices such as leveraging transfer learning, using macro-averaged metrics for imbalanced data, and employing stratified validation splits. These insights guided our selection of models and informed our overall approach to the classification task.

### <u>Benchmarking</u>
The project benchmarks include:

* __Classical Baseline Models__: Traditional machine learning models (e.g., LR, SVM, etc.) trained on TF-IDF features, offering a lightweight benchmark for comparison.
* __Recurrent Models__: Deep learning baseline models using recurrent architecture (e.g. LSTM, BiLSTM, etc.) serving as a middle ground between classical and transformer-based methods.
* __Transformer-Based Models__: Transformer models (e.g., BERT, DistilBERT, etc.) tuned to classify user-generated text into one of the seven mental health categories.

<br>

__Model Evaluation & Comparison__:
  * Quantitative metrics: Accuracy, Macro F1-score, Precision, Recall, and Confusion Matrix
  * Efficiency metrics: Training time and resource usage (i.e., FLOPS)

Here is a comparative summary of the performance across models.

Model        | Accuracy | Precision | Recall | F1 Score | Training Time | FLOPS
-------------|----------|-----------|--------|----------|---------------|---
DistilBERT   |  0.80    | 0.73      | 0.80   | 0.75     | Very High (~3K s/epoch) |
BERT         |  0.81    | 0.78      | 0.78   | 0.78     | High (~1800 s/epoch) | 
BiLSTM       |  0.77    | 0.72      | 0.69   | 0.70     | Low (~40 s/epoch) | 
LSTM         |  0.75    | 0.70      | 0.67   | 0.68     | Medium (~240 s/epoch) | 
TF-IDF + LR  |  0.76    | 0.74      | 0.67   | 0.70     | Very Low, Negligible (~21s total) | 
TF-IDF + SVM |  0.75    | 0.73      | 0.70   | 0.71     | Very Low, Negligible (~16s total) | 

### <u>Preliminary Experiments</u>

_(Section needs to be populated.)_

## 2. Model Implementation

### <u>Framework Selection</u>

In selecting frameworks and libraries for model development, we focused on aligning technical requirements with the strengths of each tool and the team’s familiarity. For deep learning models such as LSTM and BiLSTM, we chose TensorFlow and Keras due to their high-level API support, seamless GPU acceleration, and strong ecosystem for building and training recurrent neural networks. For transformer-based models like DistilBERT, we adopted the Hugging Face Transformers library, which offers pre-trained state-of-the-art models, easy-to-use tokenizers, and integration with both PyTorch and TensorFlow backends. Given our emphasis on rapid experimentation and access to domain-specific checkpoints, Hugging Face proved invaluable for fine-tuning and evaluating large language models. In parallel, for classical machine learning approaches like Logistic Regression and SVM, we used scikit-learn for its robust suite of preprocessing tools, evaluation metrics, and pipeline capabilities. The decision to use these frameworks was further informed by the team's proficiency, allowing us to develop modular, reproducible code efficiently while taking advantage of each library’s strengths in training, deployment, and experimentation.

### <u>Dataset Preparation</u>
In preparing the mental health sentiment dataset for model training and evaluation, we implemented a preprocessing and cleaning pipeline to ensure data quality and consistency. The raw dataset consisted of short text statements labeled with one of seven mental health categories (e.g., anxiety, depression, stress, bipolar, suicide, etc.). We began by removing duplicates, correcting encoding issues, and stripping irrelevant characters such as excessive punctuation or special symbols. To normalize the text, we converted all inputs to lowercase and applied tokenization tailored to the chosen models. We also handled class imbalance by analyzing label distributions and applying techniques such as stratified splitting to ensure balanced representation across training, validation, and test sets. This structured and consistent preprocessing ensured the models could learn effectively while minimizing noise and bias in the input data.

### <u>Model Development</u>

_(Section needs to be populated.)_

### <u>Training & Fine-Tuning</u>
The training and fine-tuning process for our models was tailored to the specific architecture and learning dynamics of each approach. For transformer-based models such as DistilBERT and BERT, we leveraged transfer learning by fine-tuning pre-trained weights from the Hugging Face Transformers library on our mental health sentiment dataset. This approach allowed us to benefit from rich contextual representations learned from large-scale corpora, while adjusting the models to our domain-specific task. Fine-tuning involved optimizing hyperparameters such as learning rate, batch size, number of epochs, and weight decay using stratified validation splits. For recurrent models like LSTM and BiLSTM, we experimented with different embedding dimensions, hidden units, dropout rates, and optimizers to balance performance and prevent overfitting. Classical models, including TF-IDF combined with Logistic Regression and Support Vector Machines (SVM), were trained with scikit-learn pipelines, tuning hyperparameters like regularization strength and kernel types through grid search and cross-validation. This multi-model training framework ensured each model was appropriately tuned for our classification task while allowing for fair comparisons across different methodological paradigms.

### <u>Evaluation & Metrics</u>
To evaluate the performance of our models in classifying mental health-related sentiments, we employed a suite of standard classification metrics that offer a well-rounded view of predictive effectiveness. Accuracy was used as a general indicator of overall correctness, measuring the proportion of correctly predicted labels across all classes. However, given the imbalanced nature of mental health categories, we placed greater emphasis on the macro-averaged F1-score, which equally weighs the F1-scores of all classes to ensure that minority categories like suicidal or bipolar were not overshadowed by more frequent labels. Additionally, we reported macro-averaged precision and recall to separately assess the model’s ability to minimize false positives and false negatives, respectively. This multi-metric evaluation approach ensures that our models are not only accurate but also fair and robust across all sentiment classes, which is particularly important for sensitive applications in mental health analysis.

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
Note: To get the most detailed and up-to-date repository structure, navigate to the top-level of the repository and within `bash` run `tree .`.

### Version Control & Collaboration
Our team used Git as the primary version control system to manage code development and collaboration throughout the project. We adopted a branching strategy where each team member worked on their own individual branches (e.g., `jie`, `jiajin`, `peter`, and `curtis`) and merged into the shared `main` branch via pull requests. To coordinate work we used Slack to track tasks and progress. This setup enabled asynchronous collaboration, efficient conflict resolution, and a clear history of project evolution. 

### Documentation
The README file with setup instructions is located here:
https://github.com/jiajinz/NLP_Project/blob/curtis/README.md

### Submission & Accessibility
The project is hosted in a publically accessible repository on GitHub, located at:
https://github.com/jiajinz/NLP_Project/tree/curtis