# Milestone 2: Model Development

## Research and Methods

### <u>Objectives</u>
1. <b>Transcript Segmentation & Speaker Classification</b>
  Tag each line within mixed transcripts as either “pilot” or “controller.”
1. <b>Pilot Intent & Parameter Extraction</b>
   * Within “pilot” segments, classify intent: Altitude change, heading change, clearance request, position report, etc.
   * Use sequence-labeling to pull out parameters (flight levels, headings, waypoints).
2. <b>Context Tracking</b>
   * Maintain a simple state dictionary: current altitude, heading, clearance status.
   * Update state after each pilot and generated ATC turn.
3. <b>ATC Response Generation</b>
  Fine-tune a seq-to-seq model on paired pilot-controller text to generate an ATC-compliant sentence (“Climb and maintain FL200,” “Turn right heading 090.”).

### <u>Literature Review</u>
* "Automatic Speech Recognition for Air Traffic Control" by Balakrishnan et al. 
  * Techniques to transform utterances into instructions and acknowledgments.
* "Contextualizing Air Traffic Management Conversations using Natural Language Understanding" by Rohani et al.
  * Methods for Intent Classification (IC) and Slot Filling (SF) to identify and extract Traffic Management Initiatives (TMIs) from aviation-specific dialogues. 

### <u>Benchmarking</u>
<span style="color:red">
<p>TBD</p>
<p>Proposal had something about a comparative study of classical vs. transformer-based approaches. We need to be flush this out with a description of at least one model of each type with performance results against some common dataset for comparison.</p></span>

### <u>Preliminary Experiments</u>
<span style="color:red">
<p>TBD</p>
<p>We need to identify and execute some simple experiments to satsify this requirement.</p> 
</span>

## Model Implementation

### <u>Framework Selection</u>
<span style="color:red">
<p>TBD</p>
<p>We will need an explanation as to why we selected whichever framework here.</p>
</span>

### <u>Dataset Preparation</u>
<span style="color:red">
<p>TBD</p>
<p>We need a description of the data preprocessing steps for each dataset we use here.</p>
</span>

### <u>Model Development</u>
<span style="color:red">
<p>TBD</p>
<p>We need a diagram and description of the model architecture here.</p>
</span>

### <u>Training & Fine-Tuning</u>
<span style="color:red">
<p>TBD</p>
<p>We will need a description of the training strategy and model hyperparameters here.</p>
</span>

### <u>Evaluation & Metrics</u>
<p>Model performance will be evaluated using the following:</p>

* BLEU Score
* ROUGE Score
* Intent parameter accuracy
* ATC phraseology compliance

## GitHub Repository Setup & Code Management

https://github.com/cneiderer/ie7500_group_project

### Repository Structure
<span style="color:red">
<p>We need to update this structure prior to submission.</p>
</span>

```
ie7500_group_project<br>
├──  .git<br>
├──  .gitignore<br>
├──  LICENSE<br>
├──  README.md<br>
├──  ie7500_group2_initialProjectProposal--SkyComm-AIDE.docx<br>
├──  data<br>
│   ├── \_\_init\_\_.py<br>
│   ├── mental_health_sentiment.csv
```