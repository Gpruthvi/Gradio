from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from PIL import Image
import torch
import io

app = Flask(__name__)

# Model ID for your fine-tuned LLaMA 3.2 Vision model
model_id = "Pruthvi369i/llama_3.2_vision_MedVQA"

# Load the model with optimized settings
pipe = pipeline("image-to-text", model=model_id, device_map="auto")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    image_file = request.files['image']
    question = request.form['question']
    image = Image.open(io.BytesIO(image_file.read()))
    
    # Get the answer from the model
    response = pipe(image, return_text=True)
    
    return jsonify({'answer': response[0]})

if __name__ == '__main__':
    app.run(debug=True)
