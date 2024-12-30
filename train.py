from ultralytics import YOLO
model = YOLO("yolo11n-seg.pt")  # build a new model from YAML
model.train(data='/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/config.yaml',epochs=100,imgsz=640)