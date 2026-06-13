

from ultralytics import YOLO

def load_model(model_name="yolov8n", pretrained=True):
    if pretrained:
        model = YOLO(f"{model_name}.pt")  
    else:
        model = YOLO(f"{model_name}.yaml") 
    return model

def get_model_summary(model):
    print(f"Model type: {type(model)}")
    print(f"Model info: {model.model}")
