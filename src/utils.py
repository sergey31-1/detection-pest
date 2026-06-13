
import os
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def setup_logging(log_dir="logs"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_training_plots(results, output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(results, 'history'):
        history = results.history
        plt.figure()
        plt.plot(history.get('train/loss', []), label='train loss')
        plt.plot(history.get('val/loss', []), label='val loss')
        plt.legend()
        plt.title("Loss curves")
        plt.savefig(output_dir / "loss_curves.png")
        plt.close()
    else:
        print("No history found; using YOLO's default plots.")

def calculate_metrics(results):

    return {
        'mAP50': results.box.map50,
        'mAP50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr
    }

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
