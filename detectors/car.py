# detectors/car.py
import cv2

def detect_cars(image_path, cascade_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or cannot be loaded.")

    car_classifier = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return image
