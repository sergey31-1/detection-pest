
import os
import sys
import yaml
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import load_model
from src.utils import setup_logging, save_training_plots

def train(config_path="config/train_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logging(config['paths']['outputs_dir'])
    logger.info("Starting training with config: %s", config)
    
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    model = load_model(model_name, pretrained)
    logger.info(f"Loaded model: {model_name}, pretrained={pretrained}")
    
    data_yaml = config['data']['dataset_yaml']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    imgsz = config['training']['imgsz']
    lr = config['training']['lr0']
    device = config['training']['device']
    workers = config['training']['workers']
    augment = config['training']['augment']
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        device=device,
        workers=workers,
        augment=augment,
        project=config['paths']['outputs_dir'],
        name='exp',
        exist_ok=True
    )
    
    best_weights_path = Path(config['paths']['weights_dir']) / config['paths']['best_model_name']
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    src_weights = Path(config['paths']['outputs_dir']) / 'exp' / 'weights' / 'best.pt'
    if src_weights.exists():
        shutil.copy(src_weights, best_weights_path)
        logger.info(f"Best model saved to {best_weights_path}")
    
    save_training_plots(results, config['paths']['outputs_dir'])
    
    logger.info("Training completed successfully.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
