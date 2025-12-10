# Model training script
# This script defines the model architecture and runs the training loop.
import config
import pandas
from utils import setup_logger, format_config_log, format_epoch_log
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

logger = setup_logger()

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

# 2. DATA: Create a Custom Dataset Class
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

def validate():
  model.eval()
  total_val_loss = 0
  total_correct_predictions = 0
  all_labels = []
  all_predictions = []

  with torch.no_grad():
    for batch in val_loader:
      input_ids = batch["input_ids"].to(device)
      mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      logits = model(input_ids=input_ids, attention_mask=mask)

      loss = criterion(logits, labels)
      total_val_loss += loss.item()

      # Get predictions and update total_correct_predictions
      predictions = torch.argmax(logits, dim=-1)
      total_correct_predictions += (predictions == labels).sum().item()

      # Store labels and predictions for Kappa calculation
      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predictions.cpu().numpy())


  val_loss = total_val_loss / len(val_loader)
  val_acc = total_correct_predictions / len(val_dataset) * 100  # Accuracy as a percentage

  # Calculate Quadratic Weighted Cohen Kappa score
  kappa = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')

  return val_loss, val_acc, kappa

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for training.")
        
    model_name = "SZTAKI-HLT/hubert-base-cc"
    model = LeanDeepHubert(model_name, num_labels = config.NUM_LABELS)
    logger.info(f"Loading tokenizer for model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    best_params = model.state_dict()


    logger.info("Preparing data loaders...")
    df = pandas.read_csv(config.TRAINING_DATA_PATH)
    x_train, x_val, y_train, y_val = train_test_split(df["paragraph"], df["label"], test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE_SPLIT)
    train_dataset = HungarianTextDataset(x_train.values, y_train.values, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataset = HungarianTextDataset(x_val.values, y_val.values, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    logger.info(f"Split data into training and validation sets with {len(train_dataset)} and {len(val_dataset)} samples, respectively.")


    logger.info(format_config_log(config, device))
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    weights = [656/95, 656/261, 656/439, 656/641, 656/656]

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0

        model.train()

        for batch_idx, batch in enumerate(train_loader):

            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids=input_ids, attention_mask=mask)

            loss = criterion(input=logits, target=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

            # Removed break to allow full epoch training

        avg_loss = total_loss / len(train_loader) # Corrected loader to train_loader
        logger.info(f"--> Epoch {epoch+1} Completed. Validation starting...")
        val_loss, val_acc, kappa = validate() # Updated to receive kappa score
        logger.info(format_epoch_log(epoch+1, avg_loss, val_loss, val_acc, kappa))
        if (val_loss < best_val_loss):
            best_params = model.state_dict()

        logger.info(f"Training finished. Saving the best model to {config.MODEL_SAVE_PATH} and the tokenizer to {config.TOKENIZER_SAVE_PATH}...")
        model.load_state_dict(best_params)
        tokenizer.save_pretrained(config.TOKENIZER_SAVE_PATH)
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info("Model and tokenizer saved successfully.")

