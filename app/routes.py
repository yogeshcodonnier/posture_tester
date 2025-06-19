from flask import Blueprint, jsonify, render_template, request
import torch
import sys
from pathlib import Path
from app.services import imgDetect

main = Blueprint('main', __name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_posture_new_1.pt', force_reload=False)


# Add YOLOv5 local directory to sys.path
# sys.path.append(str(Path(__file__).resolve().parent.parent / 'yolov5'))
# yolov5_root = Path(__file__).resolve().parent.parent / 'yolov5'
# sys.path.append(str(yolov5_root))

# from models.common import DetectMultiBackend
# # Load the model
# weights = Path(__file__).resolve().parent.parent / 'best_posture_new_1.pt'
# model = DetectMultiBackend(weights, device='cpu')


@main.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@main.route('/detect', methods=['POST'])
def detect():
    if 'files' not in request.files:
        return jsonify({'error': 'No files found'}), 400

    img = request.files['files']
    result = imgDetect(img,model)
    return jsonify(result)