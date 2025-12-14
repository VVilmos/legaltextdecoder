# Model training script
# This script defines the model architecture and runs the training loop.
import config
import pandas
from utils import setup_logger, format_config_log, format_epoch_log
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.optim import AdamW
from models import LeanDeepHubert, HungarianTextDataset
from sklearn.metrics import cohen_kappa_score
from torchinfo import summary

logger = setup_logger()


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

      predictions = torch.argmax(logits, dim=-1)
      total_correct_predictions += (predictions == labels).sum().item()

      all_labels.extend(labels.cpu().numpy())
      all_predictions.extend(predictions.cpu().numpy())


  val_loss = total_val_loss / len(val_loader)
  val_acc = total_correct_predictions / len(val_dataset) * 100  

  kappa = cohen_kappa_score(all_labels, all_predictions, weights='quadratic')

  return val_loss, val_acc, kappa

if __name__ == "__main__":

    logger.info(" ****************************** TRAINING STARTED ******************************")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model_name = "SZTAKI-HLT/hubert-base-cc"
    model = LeanDeepHubert(model_name, num_labels = config.NUM_LABELS)
    logger.info(summary(model, verbose=0))
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


    logger.info(f"Using device: {device} for training.")
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


        avg_loss = total_loss / len(train_loader)
        logger.info(f"--> Epoch {epoch+1} Completed. Validation starting...")
        val_loss, val_acc, kappa = validate() 
        logger.info(format_epoch_log(epoch+1, avg_loss, val_loss, val_acc, kappa))
        if (val_loss < best_val_loss):
            best_params = model.state_dict()

        logger.info(f"Training finished. Saving the best model to {config.MODEL_SAVE_PATH} and the tokenizer to {config.TOKENIZER_SAVE_PATH}...")
        tokenizer.save_pretrained(config.TOKENIZER_SAVE_PATH)
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        logger.info("Model and tokenizer saved successfully.")

