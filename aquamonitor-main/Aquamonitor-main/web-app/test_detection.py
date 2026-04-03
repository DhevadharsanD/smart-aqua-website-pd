import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import subprocess
import json

# Test with sample image
print("Testing threat detection on sample.jpg...")
result = subprocess.run(['python', 'predict_predator.py', 'sample.jpg'], 
                       capture_output=True, text=True, cwd=os.getcwd())

# Parse output - get last line which should be JSON
lines = result.stdout.strip().split('\n')
json_line = lines[-1]
try:
    detection = json.loads(json_line)
    print(f"✓ Detection result: {detection}")
    print(f"  Predator: {detection['predator']}")
    print(f"  Confidence: {detection['confidence']}")
except:
    print(f"✗ Failed to parse output: {json_line}")
    print(f"Full stderr: {result.stderr}")
