from flask import Blueprint, request, jsonify, render_template
from app.services import imgDetect
from pathlib import Path
import sys

# Add YOLOv5 path
yolov5_root = Path(__file__).resolve().parent.parent / 'yolov5'
sys.path.append(str(yolov5_root))

from models.common import DetectMultiBackend

main = Blueprint('main', __name__)

# Load model once
weights = Path(__file__).resolve().parent.parent / 'best_posture_new_1.pt'
model = DetectMultiBackend(weights, device='cpu', dnn=False)

print("model")
print(model)

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/detect', methods=['POST'])
def detect():
    if 'files' not in request.files:
        return jsonify({'status': '0', 'msg': 'No files found'}), 400

    img = request.files['files']
    result = imgDetect(img, model)
    return jsonify(result)
