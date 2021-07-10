"""
HOW TO RUN THE FLASK APP? FLASK_APP=app.py flask run
HOW TO KILL THE FLASK SERVER? Ctrl + C
HOW TO RUN HOT RELOAD FLASK APP? FLASK_APP=app.py FLASK_ENV=development flask run
0 -> with mask
1 -> without mask
"""

from flask import Flask, request, render_template, Response
from flask_socketio import SocketIO, emit
from camera import Camera
import cv2
import os
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

webApp = Flask(__name__)
socketio = SocketIO(webApp)
camera = Camera()

cfg = "./utils/tiny-yolo-widerface.cfg"
weights = "./utils/tiny-yolo-widerface_final.weights"
net = cv2.dnn.readNet(weights, cfg)


def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def valid_face_coord(face):
    if len(face) >= 4:
        if face[0] < 0:
            return False
        if face[1] < 0:
            return False
        if face[2] < 0:
            return False
        if face[3] < 0:
            return False
    return True


def openCV_draw_boundary_box(image, face, p):
    if p == 1:
        # red color === without mask
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
    else:
        # green color === with mask
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)


def openCV_image_cropping(img, face):
    crop_img = img[face[1] : face[3], face[0] : face[2]]
    return crop_img


class CModel:
    def __init__(self, file_path):
        self.model = self.read_model_from_file(file_path)

    def predict_single(self, image):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        p = self.model.predict(input_arr, verbose=0)[0]
        return np.argmax(p)

    def read_model_from_file(self, file_path):
        model = tf.keras.models.load_model(file_path)
        model.summary()
        return model


model = CModel("./utils/modelV3_mobile_net_v2_10epochs_AIZOO_rescaling.h5")


def magic(frame):
    # dl on frame
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    for out in outs:
        for face in out:
            print(face)
            scores = face[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(face[0] * frame_width)
                center_y = int(face[1] * frame_height)
                w = int(face[2] * frame_width)
                h = int(face[3] * frame_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                face_co_ordinates = [x, y, x + w, y + h]
                if valid_face_coord(face_co_ordinates):
                    cropped_face = openCV_image_cropping(frame, [x, y, x + w, y + h])
                    p = model.predict_single(cropped_face)
                    openCV_draw_boundary_box(frame, [x, y, x + w, y + h], p)
    return frame


@webApp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# @webApp.route("/video_feed", methods=["GET"])
# def video_feed():
#     fps = FPS().start()
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("input image", namespace="/classify")
def classify(input):
    input = input.split(",")[1]
    image_data = input
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    image_array = magic(image_array)
    image_bytes = bytes(Image.fromarray(image_array).tobytes())
    image_data = "data:image/jpeg;base64," + str(image_data)
    emit("out-image-event", {"image_data": image_data}, namespace="/classify")


if __name__ == "__main__":
    socketio.run(webApp)