import tf2onnx
import tensorflow as tf
import onnxruntime as ort
import numpy as np

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    # Load model
    model = tf.keras.models.load_model(keras_model_path)

    # Specify input signature (batch size set to support dynamic batch)
    spec = (tf.TensorSpec((None, 96, 96, 3), tf.float32, name="input"),)

    # Convert model
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # Save ONNX model
    with open(onnx_model_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"ONNX model saved to {onnx_model_path}")

    # Optional: Verify ONNX model
    session = ort.InferenceSession(onnx_model_path)
    print("ONNX model verification succeeded，input name:", session.get_inputs()[0].name)

if __name__ == '__main__':
    convert_keras_to_onnx('models/best_model.h5', 'models/model.onnx')