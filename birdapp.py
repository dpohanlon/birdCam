import cv2
import base64
from threading import Thread
from flask import Flask, render_template, Response, send_from_directory
from flask_socketio import SocketIO, emit
import os

from cam_interface import Camera
from cam_motion import detect_motion

camera = Camera.get_instance()

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/motion_images/<filename>')
def serve_image(filename):
    return send_from_directory('motion_images', filename)

@app.route('/get_latest_images')
def get_latest_images():

    files = os.listdir('motion_images')
    files = sorted(files, key=lambda x: os.path.getctime(os.path.join('motion_images', x)), reverse=True)
    return {'images': files[:5]}  # return names of 5 latest images


def get_frame():

    while True:

        frame = camera.get_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        yield frame

@socketio.on('connect', namespace='/live')
def test_connect():
    print('Client wants to connect to the socket')

    def send_frame():
        for frame in get_frame():
            socketio.emit('response', {'image': frame}, namespace='/live')

    socketio.start_background_task(send_frame)

if __name__ == '__main__':

    motion_thread = Thread(target=detect_motion, args = (camera,))
    motion_thread.start()

    # Debug mode causes problems by instantiating multiple cameras

    socketio.run(app, debug=False, host='0.0.0.0', allow_unsafe_werkzeug=True)
