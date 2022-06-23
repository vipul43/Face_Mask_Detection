from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from retinaface import RetinaFace
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
webApp = Flask(__name__)
webApp.config['UPLOAD_FOLDER'] = './uploads'
webApp.config['DOWNLOAD_FOLDER'] = './downloads'
socketio = SocketIO(webApp)

cfg = "./utils/tiny-yolo-widerface.cfg"
weights = "./utils/tiny-yolo-widerface_final.weights"
net = cv2.dnn.readNet(weights, cfg)


def get_outputs_names(net):
    layers_names = net.getLayerNames()
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
    if p == 0:
        # red color === without mask
        cv2.rectangle(image, (face[0], face[1]),
                      (face[2], face[3]), (255, 0, 0), 2)
    else:
        # green color === with mask
        cv2.rectangle(image, (face[0], face[1]),
                      (face[2], face[3]), (0, 255, 0), 2)


def openCV_image_cropping(img, face):
    crop_img = img[face[1]: face[3], face[0]: face[2]]
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
        return model


image_model = CModel("./utils/facemask12k_100epochs_xception_224_rescaled.h5")
stream_model = CModel("./utils/AIZOO_100epochs_mobilenetv2_224_rescaled.h5")


def magic1(frame, file_path):
    resp = RetinaFace.detect_faces(file_path)
    for r in resp:
        face = resp[r]
        if face != [] and valid_face_coord(face['facial_area']) and face["score"] > 0.95:
            cropped_face = openCV_image_cropping(
                frame, face["facial_area"])
            p = image_model.predict_single(cropped_face)
            openCV_draw_boundary_box(frame, face['facial_area'], p)
    return frame


def magic2(frame):
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
                    cropped_face = openCV_image_cropping(
                        frame, [x, y, x + w, y + h])
                    p = stream_model.predict_single(cropped_face)
                    openCV_draw_boundary_box(frame, [x, y, x + w, y + h], p)
    return frame


@webApp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@webApp.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        save_file_name = "temp." + f.filename.split('.')[1]
        file_path = os.path.join(
            webApp.config['UPLOAD_FOLDER'], secure_filename(save_file_name))
        f.save(file_path)
        image = Image.open(file_path)
        image_array = np.array(image)
        image_array = magic1(image_array, file_path)
        image = Image.fromarray(image_array)
        file_path_download = os.path.join(
            webApp.config['DOWNLOAD_FOLDER'], secure_filename(save_file_name))
        image.save(file_path_download)
        with open(file_path_download, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return jsonify({'b64image': str(encoded_string)})
    return None


@socketio.on("input image", namespace="/classify")
def classify(input):
    image_bytes = input.split(",")[1]
    image_bytes = base64.b64decode(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))  # PIL image
    image_array = np.array(image)
    image_array = magic2(image_array)
    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = base64.b64encode(buffered.getvalue())
    image_string = image_bytes.decode('utf-8')
    image_data = "data:image/jpeg;base64," + str(image_string)
    emit("out-image-event", {"image_data": image_data}, namespace="/classify")


if __name__ == "__main__":
    socketio.run(webApp, ssl_context='adhoc', host='0.0.0.0', port=8080)
