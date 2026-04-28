from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train using the single most idle GPU
results = model.train(data="ultralytics/Silkworm/data.yaml", epochs=200, imgsz=640, workers=0)

# Train using the two most idle GPUs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[-1, -1])