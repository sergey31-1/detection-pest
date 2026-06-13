
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_dataset_structure(data_yaml_path="data/agropest.yaml"):
    import yaml
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    base_path = Path(data_cfg['path'])
    for split in ['train', 'val', 'test']:
        if split in data_cfg:
            img_dir = base_path / data_cfg[split]
            if img_dir.exists():
                print(f"[OK] {split}: {img_dir}")
            else:
                print(f"[WARN] {split} directory not found: {img_dir}")
    
    train_dir = base_path / data_cfg['train']
    images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    print(f"Train images count: {len(images)}")
    return True

def visualize_annotations(image_path, label_path, class_names=None):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    if not os.path.exists(label_path):
        print(f"No label file: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h
        
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if class_names:
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
