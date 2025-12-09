# Model training script
# This script defines the model architecture and runs the training loop.
import config
import pandas
from utils import setup_logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

logger = setup_logger()

def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    
    # Simulation of training loop
    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.EPOCHS} - Training...")
    
    logger.info("Training complete.")
    
    
    
"""# huBert fine-tuning"""


class LeanDeepHubert(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Reduced hidden size from 256 to 128 for safety
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),    # Keeps data centered, helps with "twisted" distributions
            nn.ReLU(),              # Adds the curve capability
            nn.Dropout(0.4),        # Increased Dropout (0.3 -> 0.4) for extra safety
            nn.Linear(128, num_labels)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# Hyperparams
batch_size = 8
lr = 2e-5
num_epochs = 3
num_labels = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
model_name = "SZTAKI-HLT/hubert-base-cc"
model = LeanDeepHubert(model_name, num_labels = 5)
model = model.to(device)

# 4. PREPARE DATALOADER
df = pandas.read_csv("./drive/MyDrive/DL/training.csv")
df["label"] = df["label"] -1
x_train, x_test, y_train, y_test = train_test_split(df["paragraph"], df["label"], test_size=0.33, random_state=33)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=40)

train_dataset = HungarianTextDataset(x_train.values, y_train.values, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = HungarianTextDataset(x_test.values, y_test.values, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

"""### Training loop"""

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

optimizer = AdamW(model.parameters(), lr=lr)
weights = [656/95, 656/261, 656/439, 656/641, 656/656]

class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def train():
    print("Starting training...")
    best_val_loss = float('inf')

    num_epochs = 3
    for epoch in range(num_epochs):
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
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

            # Removed break to allow full epoch training

        avg_loss = total_loss / len(train_loader) # Corrected loader to train_loader
        print(f"--> Epoch {epoch+1} Complete. Average Train Loss: {avg_loss:.4f}")
        val_loss, val_acc, kappa = validate() # Updated to receive kappa score
        print(f"--> Epoch {epoch+1} Complete. Average Val Loss: {val_loss:.4f}")
        print(f"--> Epoch {epoch+1} Complete. Average Val Acc: {val_acc:.4f}")
        print(f"--> Epoch {epoch+1} Complete. Quadratic Weighted Kappa: {kappa:.4f}")

        if (val_loss < best_val_loss):
            torch.save(model.state_dict(), "./drive/MyDrive/deephubert_state.bin")


        model.load_state_dict(torch.load("./drive/MyDrive/deephubert_state.bin"))
        tokenizer.save_pretrained("./drive/MyDrive/DL/leanhubert")

if __name__ == "__main__":
    train()
