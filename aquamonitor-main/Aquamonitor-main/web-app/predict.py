# import sys
# import json
# import numpy as np
# from datetime import datetime, timedelta

# SPECIES_THRESHOLDS = {
#     'general': {'tempMin': 20, 'tempMax': 28, 'doMin': 5, 'phMin': 6.5, 'phMax': 8.5},
#     'tilapia': {'tempMin': 25, 'tempMax': 32, 'doMin': 3, 'phMin': 6, 'phMax': 9},
#     'catfish': {'tempMin': 24, 'tempMax': 30, 'doMin': 4, 'phMin': 6.5, 'phMax': 8.5},
#     'salmon': {'tempMin': 10, 'tempMax': 18, 'doMin': 7, 'phMin': 6.5, 'phMax': 8},
#     'trout': {'tempMin': 10, 'tempMax': 16, 'doMin': 7, 'phMin': 6.5, 'phMax': 8},
#     'carp': {'tempMin': 20, 'tempMax': 28, 'doMin': 4, 'phMin': 6.5, 'phMax': 9},
#     'shrimp': {'tempMin': 26, 'tempMax': 32, 'doMin': 4, 'phMin': 7, 'phMax': 8.5},
#     'prawn': {'tempMin': 26, 'tempMax': 31, 'doMin': 4, 'phMin': 7, 'phMax': 8.5}
# }

# def generate_next_hour_prediction(current_temp, current_do, current_ph):
#     """Generate realistic next-hour predictions based on current values"""
#     # Add small random variations to simulate natural changes
#     temp_change = np.random.normal(0, 0.5)  # ±0.5°C variation
#     do_change = np.random.normal(0, 0.3)    # ±0.3 mg/L variation  
#     ph_change = np.random.normal(0, 0.1)    # ±0.1 pH variation
    
#     # Apply realistic constraints
#     next_temp = max(5, min(40, current_temp + temp_change))  # Keep within 5-40°C
#     next_do = max(0.5, min(20, current_do + do_change))      # Keep within 0.5-20 mg/L
#     next_ph = max(5.0, min(10.0, current_ph + ph_change))   # Keep within 5-10 pH
    
#     return round(next_temp, 2), round(next_do, 2), round(next_ph, 2)

# try:
#     temp = float(sys.argv[1])
#     do = float(sys.argv[2])
#     ph = float(sys.argv[3])
#     species = sys.argv[4] if len(sys.argv) > 4 else 'general'
    
#     thresholds = SPECIES_THRESHOLDS.get(species, SPECIES_THRESHOLDS['general'])
    
#     # Calculate quality score for CURRENT values
#     score = 0
    
#     # Temperature scoring
#     if thresholds['tempMin'] <= temp <= thresholds['tempMax']:
#         score += 35
#     elif thresholds['tempMin'] - 5 <= temp < thresholds['tempMin'] or thresholds['tempMax'] < temp <= thresholds['tempMax'] + 5:
#         score += 20
#     else:
#         score += 5
    
#     # DO scoring
#     if do >= thresholds['doMin'] + 3:
#         score += 40
#     elif do >= thresholds['doMin']:
#         score += 25
#     else:
#         score += 10
    
#     # pH scoring
#     if thresholds['phMin'] <= ph <= thresholds['phMax']:
#         score += 25
#     elif thresholds['phMin'] - 0.5 <= ph < thresholds['phMin'] or thresholds['phMax'] < ph <= thresholds['phMax'] + 0.5:
#         score += 15
#     else:
#         score += 5
    
#     if score >= 80:
#         quality, color = "Excellent", "#4CAF50"
#     elif score >= 60:
#         quality, color = "Good", "#8BC34A"
#     elif score >= 40:
#         quality, color = "Fair", "#FFC107"
#     else:
#         quality, color = "Poor", "#F44336"
    
#     recommendations = []
#     if temp < thresholds['tempMin'] or temp > thresholds['tempMax']:
#         recommendations.append(f"Temperature ({temp}°C) outside optimal range ({thresholds['tempMin']}-{thresholds['tempMax']}°C) for {species}")
#     if do < thresholds['doMin']:
#         recommendations.append(f"Dissolved oxygen ({do} mg/L) below minimum ({thresholds['doMin']} mg/L) for {species}. Consider aeration.")
#     if ph < thresholds['phMin'] or ph > thresholds['phMax']:
#         recommendations.append(f"pH ({ph}) outside optimal range ({thresholds['phMin']}-{thresholds['phMax']}) for {species}")
#     if not recommendations:
#         recommendations.append(f"All parameters are within optimal range for {species}")
    
#     # Generate next-hour predictions (different from current values)
#     next_temp, next_do, next_ph = generate_next_hour_prediction(temp, do, ph)
#     next_hour = datetime.now() + timedelta(hours=1)
    
#     result = {
#         'quality_score': round(score, 2),
#         'quality_level': quality,
#         'color': color,
#         'recommendations': recommendations,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'predicted_values': {
#             'water_temp': next_temp,
#             'do': next_do,
#             'ph': next_ph
#         },
#         'prediction_time': next_hour.strftime('%Y-%m-%d %H:%M:%S'),
#         'current_values': {
#             'water_temp': temp,
#             'do': do,
#             'ph': ph
#         }
#     }
    
#     print(json.dumps(result))
    
# except Exception as e:
#     print(json.dumps({'error': str(e)}))
#     sys.exit(1)


#!/usr/bin/env python3
import sys
import json
import os
import warnings
import contextlib
import io
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# Predefined predator class names
class_names = {
    0: 'egret',
    1: 'heron',
    2: 'kingfisher',
    3: 'cormorant',
    4: 'duck',
    5: 'swan',
    6: 'goose',
    7: 'pelican',
    8: 'stork',
    9: 'ibis',
    10: 'spoonbill',
    11: 'crane',
    12: 'flamingo',
    13: 'tern',
    14: 'gull',
    15: 'albatross',
    16: 'penguin',
    17: 'osprey',
    18: 'eagle',
    19: 'hawk',
    # Note: Ensure these IDs match your model's actual output indices
    20: 'tortoise',
    21: 'snake'
}

try:
    import tensorflow as tf
except:
    tf = None

def get_tf_interpreter(model_path):
    if tf is not None:
        try:
            return tf.lite.Interpreter(model_path=model_path)
        except Exception as e:
            pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except Exception as e:
        raise Exception(f"Failed to create TFLite interpreter: {e}")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    if a1 + a2 - inter_area <= 0:
        return 0
    return inter_area / (a1 + a2 - inter_area)

def nms(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0: return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        others = idxs[1:]
        keep_mask = np.array([iou(boxes[i], boxes[j]) < iou_thresh for j in others])
        idxs = others[keep_mask]
    return keep

def load_image(image_path, target_size, dtype):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size, Image.BILINEAR)
    data = np.array(img_resized, dtype=np.float32)
    if dtype == np.uint8:
        data = data.astype(np.uint8)
    else:
        data = data / 255.0
    data = np.expand_dims(data, axis=0)
    return data

def parse_predictions(output, model_input_shape, score_thresh=0.25):
    if output.ndim == 3 and output.shape[2] in (6, 7):
        boxes = []
        scores = []
        classes = []
        for det in output[0]:
            conf = float(det[4])
            if conf < score_thresh:
                continue
            cls = int(det[5])
            boxes.append([float(det[0]), float(det[1]), float(det[2]), float(det[3])])
            scores.append(conf)
            classes.append(cls)
        return boxes, scores, classes

    if output.ndim == 3 and output.shape[2] >= 6:
        pred = output[0]
        boxes = []
        scores = []
        classes = []
        box_xywh = pred[:, :4]
        # Check if model output is raw logits (needs sigmoid) or already activated
        objectness = sigmoid(pred[:, 4]) if pred.shape[-1] > 5 else pred[:, 4]
        class_scores = sigmoid(pred[:, 5:])
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_scores)), class_ids]
        final_conf = objectness * class_conf
        for i in range(len(pred)):
            if final_conf[i] < score_thresh:
                continue
            cx, cy, w, h = box_xywh[i]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(float(final_conf[i]))
            classes.append(int(class_ids[i]))
        return boxes, scores, classes

    return [], [], []

def main():
    try:
        if len(sys.argv) < 2:
            raise Exception('Image path not provided')

        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            raise Exception(f'Image file not found: {img_path}')

        # Validate image
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            raise Exception(f'Invalid image file: {e}')

        # Find model
        model_candidates = [
            os.path.join(os.path.dirname(__file__), 'third_yolo.tflite'),
            os.path.join(os.path.dirname(__file__), 'model.pt'),
            os.path.join(os.path.dirname(__file__), 'best.pt', 'third_yolo.pt'),
        ]

        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if not model_path:
            raise Exception(f'Model not found')

        # Run inference (suppress output)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            interpreter = get_tf_interpreter(model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input_shape = input_details[0]['shape']
            target_size = (int(input_shape[2]), int(input_shape[1])) if len(input_shape) == 4 else (640, 640)

            input_data = load_image(img_path, target_size, input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            boxes, scores, classes = parse_predictions(output_data, input_shape, score_thresh=0.25)

        if not boxes:
            result = {'predator': 'No Predator Detected', 'confidence': 0.0}
            print(json.dumps(result))
            return

        keep = nms(np.array(boxes), np.array(scores), iou_thresh=0.45)
        if not keep:
            result = {'predator': 'No Predator Detected', 'confidence': 0.0}
            print(json.dumps(result))
            return

        top_idx = keep[0]
        top_cls = int(classes[top_idx])
        top_conf = float(scores[top_idx])

        # Get the initial name from mapping
        predator_name = class_names.get(top_cls, f"Unknown Class {top_cls}").lower()

        # --- FIX FOR SWAPPED LABELS ---
        if predator_name == 'tortoise':
            predator_name = 'snake'
        elif predator_name == 'snake':
            predator_name = 'tortoise'
        # ------------------------------

        result = {'predator': predator_name.capitalize(), 'confidence': round(top_conf, 4)}
        print(json.dumps(result))

    except Exception as e:
        result = {'predator': 'Error', 'confidence': 0.0, 'error': str(e)}
        print(json.dumps(result))

if __name__ == '__main__':
    main()