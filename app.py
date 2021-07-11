'''
HOW TO RUN THE FLASK APP? FLASK_APP=app.py flask run
HOW TO KILL THE FLASK SERVER? Ctrl + C
HOW TO RUN HOT RELOAD FLASK APP? FLASK_APP=app.py FLASK_ENV=development flask run
'''

from flask import Flask, request, render_template, jsonify
import os
import cv2
from retinaface import RetinaFace
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import base64
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def openCV_image_cropping(img, face):
    crop_img = img[face[1]: face[3], face[0]: face[2]]
    return crop_img


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


def magic1(frame, file_path):
    resp = RetinaFace.detect_faces(file_path)
    for r in resp:
        face = resp[r]
        if face != [] and valid_face_coord(face['facial_area']) and face["score"] > 0.95:
            cropped_face = openCV_image_cropping(
                frame, face["facial_area"])
            p = model.predict_single(cropped_face)
            openCV_draw_boundary_box(frame, face['facial_area'], p)
    return frame


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


webApp = Flask(__name__)
webApp.config['UPLOAD_FOLDER'] = './uploads'
webApp.config['DOWNLOAD_FOLDER'] = './downloads'


@webApp.route('/', methods=['GET'])
def test():
    return render_template('index.html')


@webApp.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']

        # Save the file to ./uploads
        save_file_name = "temp." + f.filename.split('.')[1]
        file_path = os.path.join(
            webApp.config['UPLOAD_FOLDER'], secure_filename(save_file_name))
        f.save(file_path)

        # Make prediction
        image = Image.open(file_path)
        image_array = np.array(image)
        image_array = magic1(image_array, file_path)
        image = Image.fromarray(image_array)

        # save predictions to ./downloads
        file_path_download = os.path.join(
            webApp.config['DOWNLOAD_FOLDER'], secure_filename(save_file_name))
        image.save(file_path_download)

        with open(file_path_download, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return jsonify({'b64image': str(encoded_string)})
    return None


if __name__ == "__main__":
    webApp.run(debug=True)
