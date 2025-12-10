# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger
import torch

logger = setup_logger()

def evaluate():
    logger.info("Evaluating model...")

if __name__ == "__main__":
    evaluate()

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
