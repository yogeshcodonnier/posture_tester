# from PIL import Image
# # import torch
# import io
# import math

# import torch
# import numpy as np

# # model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/xampp/htdocs/yolo_model/flask_apis/best_posture_new_1.pt', force_reload=False)

# def generate_feedback(posture_class, rating):
#     if posture_class == "Good":
#         if rating >= 9:
#             return "Perfect posture! Keep it up!"
#         elif rating >= 7:
#             return "Good posture! Keep improving!"
#         else:
#             return "Your posture is decent. Try standing taller."
#     elif posture_class == "Bad":
#         if rating >= 5:
#             return "Your posture is improving, but keep working on it."
#         else:
#             return "Poor posture. Please correct your posture by straightening your back and relaxing your shoulders."
#     else:
#         return "Posture evaluation needs improvement."

# def rate_posture(posture_class, confidence):
#     if posture_class == "Good":
#         if confidence >= 0.9:
#             return 10
#         elif confidence >= 0.8:
#             return 8
#         elif confidence >= 0.7:
#             return 7
#         else:
#             return 6
#     elif posture_class == "Bad":
#         if confidence >= 0.9:
#             return 5
#         elif confidence >= 0.8:
#             return 4
#         elif confidence >= 0.7:
#             return 3
#         else:
#             return 2
#     else:
#         return 1
    




# from utils.augmentations import letterbox  # comes with yolov5
# import cv2

# def imgDetect(img, model):
#     try:
#         # Read image
#         image = Image.open(io.BytesIO(img.read())).convert('RGB')
#         img_np = np.array(image)

#         # Letterbox resize to model expected input size
#         img_resized = letterbox(img_np, new_shape=640)[0]

#         # Convert to tensor
#         img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
#         img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

#         if torch.cuda.is_available():
#             img_tensor = img_tensor.cuda()
#             model.model = model.model.cuda()

#         # Inference
#         pred, _ = model(img_tensor, augment=False, visualize=False)

#         if pred[0].shape[0] == 0:
#             raise ValueError("No detections found.")

#         det = pred[0][0]  # First detection
#         x1, y1, x2, y2, conf, cls = det.tolist()
#         posture_class = model.names[int(cls)]

#         bbox = {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
#         posture_rating = rate_posture(posture_class, conf)
#         feedback = generate_feedback(posture_class, posture_rating)

#         return {
#             'status': '1',
#             'msg': 'Detection and posture evaluation successful',
#             'filename': img.filename,
#             'posture_class': posture_class,
#             'confidence': conf,
#             'bounding_box': bbox,
#             'posture_rating': posture_rating,
#             'feedback': feedback
#         }

#     except Exception as e:
#         return {
#             'status': '0',
#             'msg': f'Detection failed: {str(e)}'
#         }



# # def imgDetect(img, model):
# #     try:
# #         image = Image.open(io.BytesIO(img.read())).convert('RGB')
        
# #         results = model(image)

# #         detections = results.pandas().xyxy[0].to_dict(orient='records')

# #         key_points = {}
# #         posture_class = None
# #         confidence = 0.0
# #         bbox = None

# #         for detection in detections:
            
# #             if 'name' in detection:
# #                 posture_class = detection['name']
# #                 confidence = detection['confidence']
# #                 bbox = {
# #                     'xmin': detection['xmin'],
# #                     'ymin': detection['ymin'],
# #                     'xmax': detection['xmax'],
# #                     'ymax': detection['ymax']
# #                 }

# #         if posture_class is None:
# #             raise ValueError("Posture class ('Good' or 'Bad') not found in the detections.")
        
# #         posture_rating = rate_posture(posture_class, confidence)

# #         feedback = generate_feedback(posture_class, posture_rating)

# #         return {
# #             'status': '1',
# #             'msg': 'Detection and posture evaluation successful',
# #             'filename': img.filename,
# #             'detections': detections,
# #             'posture_class': posture_class,
# #             'confidence': confidence,
# #             'posture_rating': posture_rating,
# #             'bounding_box': bbox,
# #             'feedback': feedback
# #         }

# #     except Exception as e:
# #         return {
# #             'status': '0',
# #             'msg': f'Detection failed: {str(e)}'
# #         }



import sys
import io
import numpy as np
from PIL import Image
from pathlib import Path

# Add yolov5 directory to path
yolov5_path = Path(__file__).resolve().parent.parent / 'yolov5'
sys.path.append(str(yolov5_path))

from utils.augmentations import letterbox  # ✅ Needed for image preprocessing

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
    return 1

def generate_feedback(posture_class, rating):
    if posture_class == "Good":
        if rating >= 9:
            return "Excellent posture! Keep it up!"
        elif rating >= 7:
            return "Good job! Your posture looks solid. Small improvements can make it perfect."
        else:
            return "Your posture is okay. Try to stand straighter and align your shoulders."
    elif posture_class == "Bad":
        if rating >= 5:
            return "You're improving, but still need to work on straightening your posture."
        else:
            return "Posture needs attention. Straighten your back, relax shoulders, and avoid slouching."
    else:
        return "Unable to determine posture quality. Please try again."

def imgDetect(img, model):
    try:
        # Read and convert image to RGB
        image = Image.open(io.BytesIO(img.read())).convert('RGB')

        # Convert to NumPy array
        image_np = np.array(image)

        # Resize using letterbox for YOLO compatibility
        img_resized, _, _ = letterbox(image_np, new_shape=(640, 640))

        # Convert to float32
        img_resized = img_resized.astype(np.float32) / 255.0
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
        img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dim
        img_tensor = torch.from_numpy(img_resized).to('cpu')

        # Run detection
        results = model(img_tensor)
        pred = results[0]  # raw output

        detections = []
        posture_class = None
        confidence = 0.0

        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det.tolist():
                    posture_class = model.names[int(cls)]
                    confidence = float(conf)
                    detections.append({
                        'bbox': xyxy,
                        'confidence': confidence,
                        'class': posture_class
                    })
                    break  # take first detection only

        if posture_class is None:
            raise ValueError("Posture class ('Good' or 'Bad') not found in the detections.")

        posture_rating = rate_posture(posture_class, confidence)
        feedback = generate_feedback(posture_class, posture_rating)

        return {
            'status': '1',
            'msg': 'Detection and posture evaluation successful',
            'filename': img.filename,
            'posture_class': posture_class,
            'confidence': confidence,
            'posture_rating': posture_rating,
            'feedback': feedback,
            'detections': detections
        }

    except Exception as e:
        return {
            'status': '0',
            'msg': f'Detection failed: {str(e)}'
        }
