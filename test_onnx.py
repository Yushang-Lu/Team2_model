import onnxruntime as ort
import numpy as np
from PIL import Image
import sys
import os

def preprocess_image(image_path, target_size=(96, 96)):
    """Load and preprocess image to model input format"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0   # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)        # Add batch dimension
    return img_array

def predict_onnx(onnx_model_path, image_path, class_names=None):
    """Predict using ONNX model"""
    # Create inference session
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    probabilities = outputs[0][0]  # Get result for first image in batch
    
    # Get top-1 prediction
    pred_class = np.argmax(probabilities)
    confidence = probabilities[pred_class]
    
    # Show class name if provided
    if class_names and pred_class < len(class_names):
        class_name = class_names[pred_class]
    else:
        class_name = str(pred_class)
    
    print(f"Predicted class: {class_name} (index: {pred_class})")
    print(f"COnfidence: {confidence:.4f}")
    return pred_class, confidence

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test_onnx.py <onnx_model_path> <image_path>")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    img_path = sys.argv[2]
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found: {onnx_path}")
        sys.exit(1)
    if not os.path.exists(img_path):
        print(f"Error: Image file not found: {img_path}")
        sys.exit(1)
    
    # Optional: Read class names from data directory (sorted folder names)
    # If you used flow_from_directory during training, classes are sorted alphabetically
    data_dir = './data'
    if os.path.exists(data_dir):
        class_names = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
    else:
        class_names = None
    
    predict_onnx(onnx_path, img_path, class_names)