from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")
model = YOLO('yolov8n-pose.pt')

# Use the model
model.train(data="/home/cose-ia/Downloads/Schema-1.json",
            epochs=300,
            batch=8,
            amp=False)
