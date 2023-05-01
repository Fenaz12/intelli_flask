from flask import Flask, request,jsonify
from PIL import Image
from ultralytics import YOLO
from flask_cors import CORS
import json
import os


app = Flask(__name__)
CORS(app)
model = YOLO("goi_v2.pt")

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image_input' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    email = request.form.get('email')
    if not email:
        return "Please provide a valid email"
    
    image_file = request.files['image_input']
    image_file.save(os.path.join('calls/', image_file.filename))
    # Open the image using Pillow
    image = Image.open(image_file)
    print("Image received successfully")

    # Process the image with your YOLO model and return the predictions
    predictions = model(image, imgsz=640)[0]
    object_dct = {'Apple': 0, 'Banana': 0, 'Bread': 0, 'Carrot': 0, 'Tomato': 0, 'Potato': 0, 'Orange': 0}
    objects = []

    for r in predictions:
        for c in r.boxes.cls:
            objects.append(model.names[int(c)])
            print(model.names[int(c)])

    for item in objects:
        if item in object_dct:
            object_dct[item] += 1
        else:
            object_dct[item] = 1

        
    response = {
        'email': email,
        'predictions': object_dct
    }
    
    with open('predictions.json', 'w') as f:
        json.dump(response, f)
    return jsonify(response)

# Serve the predictions.json file
@app.route('/api/predictions')
def get_predictions():
    with open('predictions.json', 'r') as f:
        predictions = json.load(f)
    return predictions


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    #app.run()