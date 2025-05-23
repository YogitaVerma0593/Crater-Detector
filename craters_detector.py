from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
import os
import requests
import gdown

app = Flask(__name__)

MODEL_PATH = "best.pt"
GOOGLE_DRIVE_ID = "1lbpZXbKrgGT0-UN6xc357K_nsxqgplC3"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
model = YOLO(MODEL_PATH)    # Load model once globally

@app.route("/")
def root():
    with open("index.html") as file:
        return file.read()

@app.route("/detect", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(json.dumps(boxes), mimetype='application/json')

def detect_objects_on_image(buf):
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
    return output

if __name__ == "__main__":
    print("Server running at http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)
