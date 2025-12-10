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

    all_data = []

    logger.info("Loading JSON files from extracted folders...")
    for root, dirs, files in os.walk(extracted_dir):
        # Skip any folder named "consensus"
        dirs[:] = [d for d in dirs if d.lower() != "consensus"]

        for file in files:
            if file.lower().endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    all_data.append(json.load(f))

    # Clean up extracted files after loading
    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info(f"Loaded {len(all_data)} JSON files.")
    return all_data

def parse_label(label_str):
    match = re.search("[1-5]", label_str)
    if match:
        rank = int(match[0])
    return rank

def extract_data(all_data):
    df = pandas.DataFrame()
    texts = []
    labels = []

    logger.info("Parsing paragraphs and labels from JSON files...")
    for json_data in all_data:
        for i in range(len(json_data)):
            if not isinstance(json_data, list):
                continue
            annotations = json_data[i]["annotations"][0]["result"]
            if len(annotations) > 0:
                texts.append(json_data[i]["data"]["text"])
                label = json_data[i]["annotations"][0]["result"][0]["value"]["choices"][0]
                labels.append(parse_label(label))

    df["paragraph"] = texts
    df["label"] = labels

    logger.info(f"Parsed {len(df)} samples with paragraphs and labels.")
    return df

if __name__ == "__main__":
    data = load_json()
    df = extract_data(data)
    # df.to_csv(config.TRAINING_DATA_PATH)

    # Data transformation
    df.to_csv(config.RAW_DATA_PATH, index=False)
    logger.info(f"Raw data saved for analysis to '{config.RAW_DATA_PATH}'.")
    logger.info("Processing training data...")
    logger.info(format_config_log(config, "cuuuudaaa"))
    df["label"] = df["label"] -1
    logger.info("Transforming labels by subtracting 1 to make them zero-based.")

    logger.info("Removing duplicate paragraphs...")
    logger.info(f"Found {df['paragraph'].duplicated().sum()} duplicate paragraphs.")
    df = df.drop_duplicates(subset=['paragraph'])
    logger.info(f"Total samples after removing duplicates: {len(df)}")

    logger.info("Splitting data into training and test sets with test size 33%...")
    x_train, x_test, y_train, y_test = train_test_split(df["paragraph"], df["label"], test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE_SPLIT)
    logger.info(f"Training samples: {len(x_train)}, Test samples:  {len(x_test)}")
    df_train = pandas.DataFrame({'paragraph': x_train, 'label': y_train})
    df_test = pandas.DataFrame({'paragraph': x_test, 'label': y_test})
    # Save processed data
    df_train.to_csv(config.TRAINING_DATA_PATH, index=False)
    df_test.to_csv(config.TEST_DATA_PATH, index=False)
    logger.info(f"Preprocessed training data saved to {config.TRAINING_DATA_PATH}.")
    logger.info(f"Preprocessed test data saved to {config.TEST_DATA_PATH}.")

        





