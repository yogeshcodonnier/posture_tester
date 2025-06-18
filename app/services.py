from PIL import Image
# import torch
import io
import math

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/xampp/htdocs/yolo_model/flask_apis/best_posture_new_1.pt', force_reload=False)

def generate_feedback(posture_class, rating):
    if posture_class == "Good":
        if rating >= 9:
            return "Perfect posture! Keep it up!"
        elif rating >= 7:
            return "Good posture! Keep improving!"
        else:
            return "Your posture is decent. Try standing taller."
    elif posture_class == "Bad":
        if rating >= 5:
            return "Your posture is improving, but keep working on it."
        else:
            return "Poor posture. Please correct your posture by straightening your back and relaxing your shoulders."
    else:
        return "Posture evaluation needs improvement."

def rate_posture(posture_class, confidence):
    if posture_class == "Good":
        if confidence >= 0.9:
            return 10
        elif confidence >= 0.8:
            return 8
        elif confidence >= 0.7:
            return 7
        else:
            return 6
    elif posture_class == "Bad":
        if confidence >= 0.9:
            return 5
        elif confidence >= 0.8:
            return 4
        elif confidence >= 0.7:
            return 3
        else:
            return 2
    else:
        return 1

def imgDetect(img, model):
    try:
        image = Image.open(io.BytesIO(img.read())).convert('RGB')
        
        results = model(image)

        detections = results.pandas().xyxy[0].to_dict(orient='records')

        key_points = {}
        posture_class = None
        confidence = 0.0
        bbox = None

        for detection in detections:
            
            if 'name' in detection:
                posture_class = detection['name']
                confidence = detection['confidence']
                bbox = {
                    'xmin': detection['xmin'],
                    'ymin': detection['ymin'],
                    'xmax': detection['xmax'],
                    'ymax': detection['ymax']
                }

        if posture_class is None:
            raise ValueError("Posture class ('Good' or 'Bad') not found in the detections.")
        
        posture_rating = rate_posture(posture_class, confidence)

        feedback = generate_feedback(posture_class, posture_rating)

        return {
            'status': '1',
            'msg': 'Detection and posture evaluation successful',
            'filename': img.filename,
            'detections': detections,
            'posture_class': posture_class,
            'confidence': confidence,
            'posture_rating': posture_rating,
            'bounding_box': bbox,
            'feedback': feedback
        }

    except Exception as e:
        return {
            'status': '0',
            'msg': f'Detection failed: {str(e)}'
        }