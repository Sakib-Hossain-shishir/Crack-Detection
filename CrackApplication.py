import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model_path = r'F:\Machine Learning\ML prac\Surface Crack\crack_detection_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the image size that the model expects
IMG_SIZE = (150, 150)


def classify_image(image_path):
    """Classify the uploaded image."""
    # Load and preprocess the image
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Crack Detected"
    else:
        return "No Crack Detected"


def on_drop(event):
    """Handle the drop event."""
    # Extract the file path from the event data
    file_path = event.data
    # Clean up the file path
    if file_path.startswith('{') and file_path.endswith('}'):
        file_path = file_path[1:-1]
    file_path = file_path.replace('\\', '/')

    if os.path.isfile(file_path):
        # Display the image
        img = Image.open(file_path).resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Classify the image
        result = classify_image(file_path)
        result_label.config(text=result)
    else:
        result_label.config(text="File not found or invalid path")

# Create the main application window
root = TkinterDnD.Tk()
root.title("Crack Detection")

# Set window size and position
window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Configure drag-and-drop
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

# Create and place widgets
upload_label = tk.Label(root, text="Drag and Drop an Image Here", font=("Helvetica", 12), pady=20)
upload_label.pack(pady=10)

image_label = tk.Label(root, bg='white')
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Run the GUI main loop
root.mainloop()
