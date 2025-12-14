# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
import torch
from models import LeanDeepHubert, HungarianTextDataset
from transformers import AutoTokenizer
import pandas
import config
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from utils import format_model_performance, setup_logger
from baseline import fit_eval_baseline

logger = setup_logger()

if __name__ == "__main__":

  logger.info(" ****************************** EVALUATION STARTED ******************************")

  y_pred, y_test = fit_eval_baseline()  # it prints the results

  model = LeanDeepHubert(config.MODEL_NAME, num_labels = config.NUM_LABELS)
  model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  logger.info(f"Model parameters loaded from {config.MODEL_SAVE_PATH} for evaluation.")
  logger.info("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_SAVE_PATH)
  logger.info(f"Tokenizer loaded from {config.TOKENIZER_SAVE_PATH}.")
  model.eval()
  total_val_loss = 0
  total_correct_predictions = 0
  y_test = []
  y_pred = []

  logger.info("Preparing test dataloader...")
  df = pandas.read_csv(config.TEST_DATA_PATH)
  test_dataset = HungarianTextDataset(
      texts=df["paragraph"].tolist(),
      labels=df["label"].tolist(),
      tokenizer=tokenizer
  )
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
  logger.info("Testloader ready, starting evaluation of fine-tuned huBERT model...")
  criterion = torch.nn.CrossEntropyLoss()
  with torch.no_grad():
    for batch in test_loader:
      input_ids = batch["input_ids"].to(device)
      mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      logits = model(input_ids=input_ids, attention_mask=mask)

      loss = criterion(logits, labels)
      total_val_loss += loss.item()

      predictions = torch.argmax(logits, dim=-1)
      total_correct_predictions += (predictions == labels).sum().item()

      y_test.extend(labels.cpu().numpy())
      y_pred.extend(predictions.cpu().numpy())
      

  test_loss = total_val_loss / len(test_loader)
  test_acc = total_correct_predictions / len(test_dataset) * 100 

  kappa = cohen_kappa_score(y_test, y_pred, weights='quadratic')
  f1 = f1_score(y_test, y_pred, average="weighted")
  cm = confusion_matrix(y_test, y_pred)

  logger.info(format_model_performance(test_loss=test_loss, test_acc=test_acc, f1=f1, kappa=kappa, conf_matrix=cm))