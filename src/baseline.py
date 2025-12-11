import pandas
import re

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import logging

from utils import format_baseline_performance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
def word_count(paragraph):
    text = paragraph.strip()
    words = re.findall(r'\w+', text)
    return len(words)

def avg_word_length(paragraph):
    text = paragraph.strip()
    words = re.findall(r'\w+', text)
    num_words = len(words)
    num_chars = sum(len(w) for w in words)
    avg_word_len = num_chars / num_words if num_words > 0 else 0
    return avg_word_len

def avg_sentence_length(paragraph):
    sentences = re.split(r'(?<!\d)[.!?]+', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(1, len(sentences))
    num_words = len(re.findall(r'\w+', paragraph))

    avg_sent_len = num_words / num_sentences
    return avg_sent_len

def n_numbers(paragraph):
    return len(re.findall(r'[0-9]', paragraph))

def n_special_chars(paragraph):
    pattern = r"(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})|[”„]"
    return len(re.findall(pattern, paragraph))

"""Prepare data"""
def transform_dataset(df):
    df["word_count"] = [word_count(p) for p in df["paragraph"]]
    df["avg_word_length"] = [avg_word_length(p) for p in df["paragraph"]]
    df["avg_sentence_length"] = [avg_sentence_length(p) for p in df["paragraph"]]
    df["n_numbers"] = [n_numbers(p) for p in df["paragraph"]]
    df["n_special_chars"] = [n_special_chars(p) for p in df["paragraph"]]
    df = df.drop(columns=["paragraph"])

    return df


def fit_eval_baseline():
    df_train = pandas.read_csv("./data/training.csv")
    df_test = pandas.read_csv("./data/test.csv")

    df_train = transform_dataset(df_train)
    df_test = transform_dataset(df_test)
    features = ["word_count","avg_word_length", "n_numbers", "avg_sentence_length", "n_special_chars"]

    x_train, x_test, y_train, y_test = df_train[features], df_test[features], df_train["label"].to_numpy(), df_test["label"].to_numpy()
    scaler = StandardScaler()

    std_data = scaler.fit_transform(x_train)
    std_data.shape
    std_x_train = pandas.DataFrame(std_data, columns=x_train.columns)

    #model = LinearRegression().fit(x_train, y_train)
    clf = LogisticRegression(random_state=42, max_iter=10000, class_weight="balanced")
    clf.fit(std_x_train, y_train)
    coefficients = clf.coef_

    """### Model evaluation"""
    std_test_data = scaler.transform(x_test)
    std_x_test = pandas.DataFrame(std_test_data, columns=x_test.columns)
    y_pred = clf.predict(std_x_test)

    kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(format_baseline_performance(coefficients, f1, kappa, cm))
    return y_pred, y_test
    """
    print("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    print(f"Quadratic Weighted Kappa: {kappa:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    """
