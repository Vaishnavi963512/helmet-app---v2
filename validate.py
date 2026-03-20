from ultralytics import YOLO

# 🔹 Pretrained model
model_pre = YOLO("yolov8n.pt")
metrics_pre = model_pre.val(data="data.yaml")

print("Pretrained Model Results:")
print(metrics_pre)

# 🔹 Your trained model
model_custom = YOLO("runs/detect/train/weights/best.pt")
metrics_custom = model_custom.val(data="data.yaml")

print("\nCustom Model Results:")
print(metrics_custom)