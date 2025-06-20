from flask import Blueprint, request, jsonify, render_template
from app.services import imgDetect
from pathlib import Path
import sys
import torch

YOLOV5_PATH = Path(__file__).resolve().parent.parent / 'yolov5'
sys.path.append(str(YOLOV5_PATH))


from models.common import DetectMultiBackend

main = Blueprint('main', __name__)

weights = Path(__file__).resolve().parent.parent / 'best.pt'
model = DetectMultiBackend(weights, device='cpu', dnn=False)

# model = torch.load('best_posture_new_1.pt')['model'].float().fuse().eval()
# example_input = torch.randn(1, 3, 640, 640)
# traced_model = torch.jit.trace(model, example_input)
# traced_model.save('best_posture_scripted.pt')

# model = torch.jit.load('best_posture_scripted.pt')

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
