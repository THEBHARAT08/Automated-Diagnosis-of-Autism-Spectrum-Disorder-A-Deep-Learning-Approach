import os
from flask import Flask, request, render_template,Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv
from functools import wraps
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('InceptionV3_model.h5') 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load Haar Cascade for face detection
face_cas = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    
        file = request.files['image']
        filename = secure_filename(file.filename)
        if file:
            img_path = os.path.join('static/uploads', file.filename)
            file.save(img_path)
            
            
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)
            
            if prediction > 0.015:
                result = 'Non Autistic'
            else:
                result = 'Autistic'
            return render_template('index.html', img_path=img_path,filename=filename, result=result)
    
    return render_template('index.html', img_path=None, result=None)








@app.route('/video')
#@requires_auth
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv.VideoCapture(0)  # Use the correct camera index
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale for face detection
        bw_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(bw_img, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            (x, y, w, h) = largest_face

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_img = frame[y:y + h, x:x + w]
            face_resized = cv.resize(face_img, (256, 256))
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            pred = model.predict(face_resized)

            label = "Autism Detected" if pred > 0.15 else "No Autism"
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if pred > 0.5 else (0, 0, 255), 2)

        ret, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


        # Exit if 'x' is pressed
        if cv.waitKey(10) & 0xFF == ord("x"):
            break
    cap.release()
  
@app.route('/upload_video', methods=['POST'])
#@requires_auth
def upload_video():
    if 'video' not in request.files:
        return "No video part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400
    file.save(f'uploads/{file.filename}')
    return "Video uploaded successfully", 200






if __name__ == '__main__':
    app.run(debug=True)
