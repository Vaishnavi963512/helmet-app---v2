from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("best.pt")

results = model("test.jpg")

plt.imshow(results[0].plot())
plt.axis("off")