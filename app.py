from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load model
model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", warning="No file selected")

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        raw_output = os.path.join(OUTPUT_FOLDER, "raw.mp4")
        final_output = os.path.join(OUTPUT_FOLDER, "final.mp4")

        file.save(input_path)

        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

        # Save temporary video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_output, fourcc, fps, (width, height))

        violation_count = 0
        no_helmet_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔥 FIX: Convert 4-channel → 3-channel
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame)
            annotated_frame = results[0].plot()

            labels = results[0].boxes.cls.tolist()

            # 🚨 SMART VIOLATION LOGIC
            if 0 not in labels:  # 0 = helmet
                no_helmet_frames += 1

                if no_helmet_frames > 10:
                    violation_count += 1
                    no_helmet_frames = 0

                text = "NO HELMET!"
                color = (0, 0, 255)
            else:
                no_helmet_frames = 0
                text = "SAFE"
                color = (0, 255, 0)

            # Display text on video
            cv2.putText(
                annotated_frame,
                text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

            out.write(annotated_frame)

        cap.release()
        out.release()

        # 🔥 Convert to browser-compatible video (IMPORTANT)
        os.system(f"ffmpeg -y -i {raw_output} -vcodec libx264 {final_output}")

        # 🔊 Alarm trigger logic
        if violation_count > 0:
            warning = "⚠️ NO HELMET DETECTED!"
        else:
            warning = "✅ SAFE"

        return render_template(
            "index.html",
            video=final_output,
            violations=violation_count,
            warning=warning
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)