#!/usr/bin/env python3
"""
Predator Detection using YOLO Model
Detects 6 threat classes: cormorant, egret, heron, tortoise, snake, human intruder
"""

import sys
import json
import os
import warnings

warnings.filterwarnings('ignore')

def main():
    try:
        # Validate inputs
        if len(sys.argv) < 2:
            print(json.dumps({'error': 'Image path not provided', 'predator': 'Error', 'confidence': 0}))
            return

        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(json.dumps({'error': f'Image file not found: {image_path}', 'predator': 'Error', 'confidence': 0}))
            return

        # Import YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            print(json.dumps({'error': 'YOLO not installed', 'predator': 'Error', 'confidence': 0}))
            return

        # Load model - suppress output
        import io
        import contextlib
        
        # Find model file
        model_path = None
        candidates = [
            os.path.join(os.path.dirname(__file__), 'best.pt', 'best.pt'),     # Inside best.pt folder
            os.path.join(os.path.dirname(__file__), 'best.pt', 'third_yolo.pt'), 
            os.path.join(os.path.dirname(__file__), 'third_yolo.tflite'),
            os.path.join(os.path.dirname(__file__), 'model.pt'),
            os.path.join(os.path.dirname(__file__), 'best.pt'),
        ]
        
        for candidate in candidates:
            if os.path.isfile(candidate):
                model_path = candidate
                break
        
        if not model_path:
            print(json.dumps({'error': 'Model not found', 'predator': 'Error', 'confidence': 0}))
            return

        # Load YOLO model with suppressed output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = YOLO(model_path)
        
        # Get class names from model
        class_names = model.names
        print(f"Model classes: {class_names}", file=sys.stderr)
        
        # Run inference with suppressed output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = model(image_path, conf=0.25)
        
        # Process results
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    cls_name = class_names.get(cls_id, f"Class {cls_id}")
                    detections.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': confidence
                    })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not detections:
            result = {'predator': 'No Threat Detected', 'confidence': 0.0}
        else:
            top_detection = detections[0]
            result = {
                'predator': top_detection['class_name'],
                'confidence': round(top_detection['confidence'], 4)
            }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'error': str(e), 'predator': 'Error', 'confidence': 0}))

if __name__ == '__main__':
    main()
