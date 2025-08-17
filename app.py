from flask import Flask, render_template, request, send_file
import cv2
import os
from detectors import face, eye, car, pedestrian

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Ensure folders exist
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    detection_type = request.form.get('detection_type')

    if not image or not detection_type:
        return "Missing image or detection type", 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        # Apply selected detection
        if detection_type == 'face':
            output = face.detect_faces(
                image_path,
                r'C:\Users\HP\VS Code Projects\OPENCV\VisionGuard_AI\haarcascades\haarcascade_frontalface_default.xml'
            )
        elif detection_type == 'eye':
            output = eye.detect_eyes(
                image_path,
                r'C:\Users\HP\VS Code Projects\OPENCV\VisionGuard_AI\haarcascades\haarcascade_frontalface_default.xml',
                r'C:\Users\HP\VS Code Projects\OPENCV\VisionGuard_AI\haarcascades\haarcascade_eye.xml'
            )
        elif detection_type == 'car':
            output = car.detect_cars(
                image_path,
                r'C:\Users\HP\VS Code Projects\OPENCV\VisionGuard_AI\haarcascades\haarcascade_car.xml'
            )
        elif detection_type == 'pedestrian':
            output = pedestrian.detect_pedestrians(
                image_path,
                r'C:\Users\HP\VS Code Projects\OPENCV\VisionGuard_AI\haarcascades\haarcascade_fullbody.xml'
            )
        else:
            return "Unsupported detection type", 400

        # Save and return annotated image
        output_path = os.path.join(STATIC_FOLDER, 'output.jpg')
        cv2.imwrite(output_path, output)
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Detection error: {e}")
        return "Error processing the image. Please try again.", 500

if __name__ == '__main__':
    app.run(debug=True)

