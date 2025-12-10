# Utility functions
# Common helper functions used across the project.
import logging
import sys

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def format_config_log(config, device):
    config_str = (
        f"Starting Training Session for {config.MODEL_NAME}\n"
        f"--------------------------------------------------\n"
        f"   > Optimizer:       {config.OPTIMIZER}\n"
        f"   > Loss Function:   {config.LOSS_FUNCTION}\n"
        f"   > Learning Rate:   {config.LEARNING_RATE}\n"
        f"   > Batch Size:      {config.BATCH_SIZE}\n"
        f"   > Dropout:         {config.DROPOUT_RATE}\n"        
        f"   > Epochs:          {config.NUM_EPOCHS}\n"
        f"   > Device:          {device}\n"
        f"--------------------------------------------------"
    )
    return config_str

def format_epoch_log(epoch, train_loss, val_loss, val_acc, kappa):
    epoch_str = (
        f"Validation Results for Epoch {epoch} \n"
        f"--------------------------------------------------\n"
        f"   > Average Train Loss:      {train_loss:.4f}\n"
        f"   > Average Val Loss:        {val_loss:.4f}\n"
        f"   > Average Val Acc:         {val_acc:.4f}\n"
        f"   > Quadratic Weighted Kappa: {kappa:.4f}\n"
        f"--------------------------------------------------"
    )
    return epoch_str

def load_config():
    pass
