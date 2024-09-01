from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import winsound
import smtplib

app = Flask(__name__)

# Initialize YOLO model
model_path = r'C:/Users/HP\Desktop/main_project/besty.pt'
model = YOLO(model_path)

violence_detected = False

# Set up email parameters
EMAIL = "your_email@gmail.com"
PASSWORD = "your_password"
RECIPIENT_EMAIL = "recipient_email@example.com"

def send_alert_email():
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:
        connection.login(user=EMAIL, password=PASSWORD)
        connection.sendmail(
            from_addr=EMAIL,
            to_addrs=RECIPIENT_EMAIL,
            msg="Subject:Violence Alert!\n\nViolence has been detected. Please check the surveillance feed immediately."
        )

def detect_objects():
    global violence_detected
    cap = cv2.VideoCapture(1)
    email_sent = False  # To ensure email is sent only once per detection
    while True:
        _, frame = cap.read()
        results = model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
        frame = annotator.result()
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Check for violence prediction
        has_violence = any(box.cls != 0 and model.names[int(box.cls)] not in ["non_violence", "violence_level1"] for r in results for box in r.boxes)
        if has_violence and not email_sent:
            winsound.Beep(500, 200)  # Emit a sound
            send_alert_email()
            violence_detected = True
            email_sent = True  # Set flag to prevent multiple emails
        elif not has_violence:
            violence_detected = False
            email_sent = False  # Reset email flag when violence stops
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', violence_detected=violence_detected)

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
