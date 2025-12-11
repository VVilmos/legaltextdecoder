# Deep Learning Class (VITMMA19) Project 

### Data Preparation

All data preparation steps are implemented in script *src/01-data-preprocessing.py* script. The process consits of the following steps:
- Downloading the ZIP archive from [BME sharepoint](https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1).
- Extracting its content to a temporary directory
- Iterating through all its subdirectories and loading all JSON files. The ones located in *legaltextdecoder/consensus* directory are loaded to a separate collection.
- Parsing the texts/paragraphs and the labels from the JSON files:
- First, the consensus paragraphs are parsed and their labels are calculated from the group average, then rounded.
- Second, the rest of the paragraphs (non-consensus) are parsed. If a paragraph is already in the consensus dataset, it is dropped.

- At this stage, the raw data is saved to path *./data/raw_data.csv*
- Dropping the duplicates from all datasets (Note: it is only relevant in the non-consensus dataset, as there were paragraphs uploaded in separate and merged collections as well by some students.)
- Decreasing all labels by 1 to help loss calculation based on output logits
- Saving non-consensus dataset as training -and validaton- dataset, and saving consensus dataset as test dataset to path *app/data/training.csv* and *app/data/test.csv*, respectively.
 
### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [ ] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Vörös Vilmos
- **Aiming for +1 Mark**: No

### Solution Description

Problem is classifying paragraphs from legal documents (more precisely General Terms & Conditions) based on their understandability on a scale from 1 to 5. 1 represents "uttery difficult to interpret", while 5 means "easy to comprehend". The baseline model is a logistic regression model with the following features:
  - Average sentence length of the paragraph
  - Average word lenght
  - Length in words
  - Number numbers and special characters in the paragraph 

The solution is based on a pre-trained transformer model called [huBert](https://huggingface.co/SZTAKI-HLT/hubert-base-cc) which was blablabala. Two FC layers are trained alongside the pre-trained one to classify input paragraphs.

Loss function, optimizer

Evaluation metrics: Recall on rare labels, [Quadratic Cohen cappa score](https://en.wikipedia.org/wiki/Cohen%27s_kappa) and confusion matrix???

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. 

### File Structure and Functions


The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `models.py`: Custom architecture and dataset definiton used throughout the project.
    - `baseline.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
