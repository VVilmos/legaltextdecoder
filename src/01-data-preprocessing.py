import os
import json
import glob
import pandas
import re



def load_json():
    folder_path = "labeled_data"
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    all_data = []
    print(len(json_files))

    for file in json_files:
        with open(file, "r", encoding='utf-8') as f:
            all_data.append(json.load(f))

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
    for json_data in all_data:
        for i in range(len(json_data)):
            if not isinstance(json_data, list):
                continue
            print("yay")
            annotations = json_data[i]["annotations"][0]["result"]
            if len(annotations) > 0:
                texts.append(json_data[i]["data"]["text"])
                label = json_data[i]["annotations"][0]["result"][0]["value"]["choices"][0]
                labels.append(parse_label(label))

    df["paragraph"] = texts
    df["label"] = labels

    return df

if __name__ == "__main__":
    data = load_json()
    df = extract_data(data)
    df.to_csv("training.csv")

        





