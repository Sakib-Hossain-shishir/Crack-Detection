import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('crack_detection_model.h5')

def predict_crack(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    if prediction[0] > 0.5:
        return "Crack Detected"
    else:
        return "No Crack Detected"

# Example usage
result = predict_crack(r"C:\Users\HP\Desktop\Machine Learning\ML prac\Surface Crack\12.jpg ")
print(result)
