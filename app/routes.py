from flask import Blueprint, jsonify, request, render_template
from .services import imgDetect
import torch

main = Blueprint('main', __name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/xampp/htdocs/yolo_model/posture_tester/best_posture_new_1.pt', force_reload=False)

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
