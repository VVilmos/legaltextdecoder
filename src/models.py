from utils import setup_logger
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset
import logging


logger = logging.getLogger(__name__)

class LeanDeepHubert(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        logger.info(f"Fetching pre-trained huBERT model from Hugging Face as {model_name}...")
        self.bert = AutoModel.from_pretrained(model_name)

        # Reduced hidden size from 256 to 128 for safety
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),    # Keeps data centered, helps with "twisted" distributions
            nn.ReLU(),              # Adds the curve capability
            nn.Dropout(0.4),        # Increased Dropout (0.3 -> 0.4) for extra safety
            nn.Linear(128, num_labels)
        )
        logger.info(f"Classification head initialized with hidden size 128 and dropout 0.4, consisting of two Fully Connected layers.")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

class HungarianTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the single text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length", # Pads to max_len
            max_length=self.max_len,
            return_tensors="pt"   # Returns PyTorch tensors
        )

        # Return dictionary with squeezed tensors (remove batch dim 1)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }