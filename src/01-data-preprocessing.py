import os
import json
import glob
import pandas
import re
import tempfile
import zipfile
import shutil
import requests
import config
from utils import setup_logger, format_config_log
from sklearn.model_selection import train_test_split
import numpy

logger = setup_logger()


def _download_and_extract(url: str, target_dir: str) -> str:
    """Download a zip archive from url and extract into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
        archive_path = tmp.name

    logger.info("ZIP archive downloaded, extracting...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)

    logger.info("Extraction completed.")
    os.remove(archive_path)
    return target_dir


def load_json():
    """Download dataset zip from config.DATA_URL and load all JSON files except under consensus."""
    logger.info("Downloading and extracting data...")
    temp_dir = tempfile.mkdtemp(prefix="ltdata_")
    extracted_dir = _download_and_extract(config.DATA_URL, temp_dir)

    other_data = []
    consensus_data = []
    

    logger.info("Loading JSON files from extracted folders...")
    consensus_path = os.path.join(extracted_dir, "legaltextdecoder", "consensus")
    for root, dirs, files in os.walk(extracted_dir):

        # Separating consensus and non-consensus files
        if (os.path.samefile(root, consensus_path)):
            for file in files:
                if file.lower().endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        consensus_data.append(json.load(f))

        else:
            for file in files:
                if file.lower().endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        other_data.append(json.load(f))



    # Clean up extracted files after loading
    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"Loaded {len(other_data)} simple and {len(consensus_data)} consensus JSON files.")
    return consensus_data, other_data

def parse_label(label_str):
    match = re.search("[1-5]", label_str)
    if match:
        rank = int(match[0])
    return rank

def extract_data_C(consensus_data, other_data):
    texts = []
    label_counts = []
    label_sums = []

    logger.info("Parsing paragraphs and labels from JSON files...")
    
    logger.info("Parsing paragraph in consensus dataset...")
    for json_data in consensus_data:
        for i in range(len(json_data)):
            if not isinstance(json_data, list):
                continue
            annotations = json_data[i]["annotations"][0]["result"]
            if len(annotations) > 0:
                paragraph = json_data[i]["data"]["text"]
                label = json_data[i]["annotations"][0]["result"][0]["value"]["choices"][0]
                if paragraph in texts:
                    idx = texts.index(paragraph)
                    label_counts[idx] += 1
                    label_sums[idx] += parse_label(label)
                else:
                    texts.append(paragraph)
                    label_sums.append(parse_label(label))
                    label_counts.append(1)


    logger.info("Averaging labels of consensus paragraphs...")
    label_sums = numpy.array(label_sums)
    label_counts = numpy.array(label_counts)
    labels = (label_sums/label_counts).round()
    df_con = pandas.DataFrame()
    df_con["paragraph"] = texts
    df_con["label"] = labels


    logger.info(f"Parsed {df_con.shape[0]} Consensus paragraphs in total.")


    other_labels = []
    other_texts = []
    for json_data in other_data:
        for i in range(len(json_data)):
            if not isinstance(json_data, list):
                continue
            annotations = json_data[i]["annotations"][0]["result"]
            if len(annotations) > 0:
                paragraph = json_data[i]["data"]["text"]
                label = json_data[i]["annotations"][0]["result"][0]["value"]["choices"][0]
                if paragraph not in texts:
                    other_texts.append(paragraph)
                    other_labels.append(parse_label(label))

    df_other = pandas.DataFrame()
    df_other["paragraph"] = other_texts
    df_other["label"] = other_labels

    logger.info(f"Parsed {df_other.shape[0]} non-consensus paragraphs in total.")
    return df_con, df_other

if __name__ == "__main__":
    logger.info(" ****************************** DATA PREPROCESSING STARTED ******************************")
    consensus_data, other_data = load_json()
    df_con, df_other = extract_data_C(consensus_data=consensus_data, other_data=other_data)

    df = pandas.concat([df_con, df_other])
    # Data transformation
    df.to_csv(config.RAW_DATA_PATH, index=False)
    logger.info(f"Raw data saved for analysis to path '{config.RAW_DATA_PATH}'.")
    logger.info("Processing training data...")
    df_con["label"] = df_con["label"] -1
    df_other["label"] = df_other["label"] -1
    logger.info("Transforming labels by subtracting 1 to make them zero-based.")

    logger.info("Removing duplicate paragraphs...")
    # consensus
    logger.info(f"Found {df_con['paragraph'].duplicated().sum()} duplicate paragraphs among consensus.")
    df_con = df_con.drop_duplicates(subset=['paragraph'])
    logger.info(f"Total consensus samples after removing duplicates: {df_con.shape[0]}")

    # non-consensus
    logger.info(f"Found {df_other['paragraph'].duplicated().sum()} duplicate paragraphs among non-consensus.")
    df_other = df_other.drop_duplicates(subset=['paragraph'])
    logger.info(f"Total non-consensus samples after removing duplicates: {df_other.shape[0]}")

    logger.info("Saving training data and test data. Test dataset only consists of the paragraphs from consensus!")
    logger.info(f"Training samples: {df_other.shape[0]}, Test samples:  {df_con.shape[0]}")
    # Save processed data

    df_other.to_csv(config.TRAINING_DATA_PATH, index=False)
    df_con.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info(f"Preprocessed training data saved to path {config.TRAINING_DATA_PATH}.")
    logger.info(f"Preprocessed test data saved to path {config.TEST_DATA_PATH}.")

        





