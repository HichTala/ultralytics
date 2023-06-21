from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Use the model
model.train(data="/home/cose-ia/Downloads/YuGiOh_YOLO.v2i.yolov8/data.yaml",
            epochs=300,
            batch=8,
            amp=False)
