import os

# Limit the number of threads used by various libraries
# Now import TensorFlow and other modules
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
#import logging
from flask_ngrok import run_with_ngrok
# Initialize Flask app and CORS
app = Flask(__name__)
# Load your model
model = tf.keras.models.load_model('C:\\Users\\USER\\Desktop\\x\\g.h5')
if model!=None:
    print("Model loaded successfully!")
# Define classes and image size
classes = ['closeeye', 'openeye']
img_size = (224, 224)
# Define a function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, img_size)
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
@app.route('/', methods=['POST'])
def classify_eye_state():
    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    #image = cv2.imread(file) 
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    return jsonify({'class': predicted_class})
if __name__ == "__main__":
    app.run(host='0.0.0.0')
