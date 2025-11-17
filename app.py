import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
IMG_SIZE = (224, 224)
class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]


with tf.device('/cpu:0'):
    try:
        
        best_model = tf.keras.models.load_model('final_optimized_waste_model.h5')
        print("Model loaded successfully onto CPU!")
    except Exception as e:
        print(f"Error: Could not load model. Ensure the .h5 file is in the same folder. {e}")
        import sys
        sys.exit(1)


def classify_waste_with_sliding_window(input_image):
    img = input_image.convert('RGB')
    width, height = img.size
    window_size = 224
    stride = 112 
    all_predictions = []

    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            window = img.crop((x, y, x + window_size, y + window_size))
            
            window_array = np.array(window, dtype=np.float32) / 255.0
            window_array = np.expand_dims(window_array, axis=0) 
            
            prediction = best_model.predict(window_array, verbose=0)[0]
            max_confidence = np.max(prediction)
            
            if max_confidence > 0.50:
                all_predictions.append(prediction)
    
  
    if not all_predictions:
        return {class_names[i]: (1 / len(class_names)) for i in range(len(class_names))} 
    
    final_avg_prediction = np.mean(all_predictions, axis=0)
    
    
    confidences = {class_names[i]: float(final_avg_prediction[i]) for i in range(len(class_names))}
    return confidences


iface = gr.Interface(
    fn=classify_waste_with_sliding_window,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="🗑️ Optimized Waste Classification System (92.12% Accurate)",
    description="Upload an image (even cluttered ones) to classify waste using the Fine-Tuned MobileNetV2 with Sliding Window analysis.",
)
iface.launch()