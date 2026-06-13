
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
from src.utils import calculate_metrics, plot_confusion_matrix

def evaluate(weights_path, data_yaml, output_dir="outputs"):
 
    model = YOLO(weights_path)
    
    results = model.val(
        data=data_yaml,
        batch=16,
        imgsz=640,
        conf=0.25,
        iou=0.6,
        device='cuda',
        plots=True,
        save_json=True
    )
    
    print("\n===== Evaluation Results =====")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_metrics.txt", 'w') as f:
        f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
        f.write(f"Precision: {results.box.mp:.4f}\n")
        f.write(f"Recall: {results.box.mr:.4f}\n")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="data/agropest.yaml", help="Data config file")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()
    
    evaluate(args.weights, args.data, args.output)
