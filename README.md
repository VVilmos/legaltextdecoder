# Deep Learning Class (VITMMA19) Project 

### Data Preparation

All data preparation steps are implemented in script named *src/01-data-preprocessing.py*. The process consits of the following steps:
- Downloading the ZIP archive from [BME sharepoint](https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1).
- Extracting its content to a temporary directory
- Iterating through all its subdirectories and loading all JSON files. The ones located in *legaltextdecoder/consensus* directory are loaded to a separate collection.
- Parsing the texts/paragraphs and the labels from the JSON files:
    - First, the consensus paragraphs are parsed and their labels are calculated from the group average, then rounded.
    - Second, the rest of the paragraphs (non-consensus) are parsed. If a paragraph is already in the consensus dataset, it is dropped.

- At this stage, the raw data is saved to path *./data/raw_data.csv*
- Dropping the duplicates from all datasets (Note: it is only relevant in the non-consensus dataset, as there were paragraphs uploaded in separate and merged collections as well by some students.)
- Decreasing all labels by 1 to help loss calculation based on output logits.
- Saving non-consensus dataset as training -and validaton- dataset, and saving consensus dataset as test dataset to path *app/data/training.csv* and *app/data/test.csv*, respectively.
 
## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Vörös Vilmos
- **Aiming for +1 Mark**: No

### Solution Description

Problem is classifying paragraphs from legal documents (more precisely from General Terms & Conditions documents) based on their understandability on a scale from 1 to 5. 1 represents "uttery difficult to interpret", while 5 means "easy to comprehend". The baseline model is a logistic regression model with the following features:
  - Average sentence length of the paragraph
  - Average word lenght
  - Length in words
  - Number numbers and special characters in the paragraph 

The solution is based on a pre-trained transformer model called [huBERT](https://huggingface.co/SZTAKI-HLT/hubert-base-cc). It has the same architecture as the [BERT-base model](https://arxiv.org/abs/1810.04805) but it was specifically pre-trained on the hungarian language. More precisly, it was training on a snapshot of the Hungarian Wikipedia and on the [Hungarian Webcorpus 2.0](https://hlt.bme.hu/hu/resources/webcorpus2). The BERT-base model contains 12 stacked Transformer Encoder blocks with a hidden dimension of 768, each block having an attention layer with 12 attention heads. Futhermore, a classification head is appended to the last encoder block consisting of a hidden layer of 128 neurons and an output layer of 5 units. The classification head is trained alongside the pre-trained part. In total, this model has ~110.099k parameters and 256 non-trainable parameters corresponding to running statistics in the BatchNorm layer of the classification head.

For training the [AdamW](https://arxiv.org/abs/1711.05101) optimization algorithm is employed which minimizes the Cross-Entropy Loss as objective function. Regularization is only employed through dropout. There is no futher regularization penalty term in the objective function.

For evaluation, the project measures the performance of the baseline and the deep learing model on the following metrics: F1 score,  [Quadratic Cohen Kappa score](https://en.wikipedia.org/wiki/Cohen%27s_kappa). It also logs the confusion matrix to visualize the ability of the model to distinguish different understandibility-levels.

Concerning the methodology, the model is trained through 3 epochs. After 3 epochs, the model parameters which achieved the lowest validation lost are saved. In the evaluation phase, the performances of the baseline and the fine-tuned huBERT model are evaluted on the test dataset (which consists of the consensus paragraph). At last, the model predicts the understandability score of a new input paragraph.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. 

```bash
docker run dl-project
```

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
