# Utility functions
# Common helper functions used across the project.
import logging
import sys

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    return logging.getLogger(name)

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


def format_baseline_log():
    """
    Returns a string summarizing the baseline model configuration and feature set.
    """
    # Shortened feature names
    features = [
        "Word_Cnt",      # Word count
        "Avg_Sent_Len",  # Average sentence length
        "Avg_Word_Len",  # Average word_length
        "Num_Digits",    # Number of numbers
        "Num_Spec_Char"  # Number of special characters
    ]
    lines = [" "]
    lines.append("-" * 50)
    lines.append(" BASELINE MODEL CONFIGURATION AND REPORT")
    lines.append("-" * 50)
    lines.append(f" Model Type:    Logistic Regression")
    lines.append(f" Class Weights: Balanced")
    lines.append(f" Classes:       5 (Labels 1-5)")
    lines.append(f" Input Features ({len(features)}):")
    lines.append(f"   {', '.join(features)}")
    lines.append("-" * 50)
    
    return "\n".join(lines)

def format_baseline_performance(coeffs, f1, kappa, conf_matrix):
    # Feature names (shortened)
    features = [
        "Word_Cnt",      # Word count
        "Avg_Sent_Len",  # Average sentence length
        "Avg_Word_Len",  # Average word_length
        "Num_Digits",    # Number of numbers
        "Num_Spec_Char"  # Number of special characters
    ]
    lines = [" "]
    lines.append("-" * 50)
    lines.append(" BASELINE MODEL CONFIGURATION AND REPORT")
    lines.append("-" * 50)
    lines.append(f" Model Type:    Logistic Regression")
    lines.append(f" Class Weights: Balanced")
    lines.append(f" Classes:       5 (Labels 1-5)")
    lines.append(f" Input Features ({len(features)}):")
    lines.append(f"   {', '.join(features)}")
    feature_names = ["W_Cnt", "Avg_Sent", "Avg_Word", "Num_Nums", "Num_Spec"]
    classes = [1, 2, 3, 4, 5]

    # --- 1. Top Level Metrics ---
    lines.append(f" F1 Score (Weighted):       {f1:.4f}")
    lines.append(f" Quadratic Weighted Kappa:  {kappa:.4f}")
    lines.append("-" * 60)

    # --- 2. Coefficients Table ---
    # Shape is (n_classes, n_features). We create a table to show relations.
    lines.append(" LEARNED COEFFICIENTS (Class vs Feature):")
    
    # Header row for features
    header = f" {'Class':<6} |" + "".join([f" {f:>9} |" for f in feature_names])
    lines.append(header)
    lines.append(" " + "-"*6 + "-|-" + "-|-".join(["-"*9 for _ in feature_names]) + "-|")

    # Data rows
    for i, class_label in enumerate(classes):
        row_coeffs = coeffs[i] if i < len(coeffs) else []
        # Format each coefficient to 2 decimal places
        vals = "".join([f" {c:>9.2f} |" for c in row_coeffs])
        lines.append(f" {class_label:<6} |{vals}")
    
    lines.append("-" * 60)

    # --- 3. Confusion Matrix ---
    lines.append(" CONFUSION MATRIX (Rows=True, Cols=Predicted):")
    
    # Header for predicted labels
    # We use a width of 8 for the matrix counts
    col_headers = "".join([f" {c:>8}" for c in classes])
    lines.append(f" {'':<8} {col_headers}")
    lines.append(" " + "-" * (8 + 9 * 5))

    # Matrix rows
    for i, row in enumerate(conf_matrix):
        true_label = classes[i]
        row_str = "".join([f" {val:>8d}" for val in row])
        lines.append(f" {true_label:<8} |{row_str}")

    lines.append("="*60)
    
    return "\n".join(lines)


def format_model_performance(test_loss, test_acc, f1, kappa, conf_matrix):
    classes = [1, 2,3 ,4 , 5]
    lines = []
    lines.append("\n" + "="*60)
    lines.append(" huBERT MODEL PERFORMANCE REPORT")
    lines.append("="*60)
    
    # --- 1. Top Level Metrics ---
    lines.append(f" Test Loss:       {test_loss:.4f}")
    lines.append(f" Test Acc:       {test_acc:.4f}")
    lines.append(f" F1 Score (Weighted):       {f1:.4f}")
    lines.append(f" Quadratic Weighted Kappa:  {kappa:.4f}")
    lines.append("-" * 60)

    # --- 3. Confusion Matrix ---
    lines.append(" CONFUSION MATRIX (Rows=True, Cols=Predicted):")
    
    # Header for predicted labels
    # We use a width of 8 for the matrix counts
    col_headers = "".join([f" {c:>8}" for c in classes])
    lines.append(f" {'':<8} {col_headers}")
    lines.append(" " + "-" * (8 + 9 * 5))

    # Matrix rows
    for i, row in enumerate(conf_matrix):
        true_label = classes[i]
        row_str = "".join([f" {val:>8d}" for val in row])
        lines.append(f" {true_label:<8} |{row_str}")

    lines.append("="*60)
    
    return "\n".join(lines)

def load_config():
    pass
