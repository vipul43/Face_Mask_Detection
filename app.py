from flask import Flask, request, send_file
import os
import random
import cv2 
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import tensorflow as tf
from keras.models import model_from_json
from werkzeug.utils import secure_filename
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def retinaFace_detector(image_path, detector):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #detector accepts only RGB images
    faces = detector.predict_jsons(image)
    return faces

def openCV_image_cropping(image_path, face):
    img = cv2.imread(image_path)
    crop_img = img[face[1]:face[3], face[0]:face[2]]
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
    if p==1:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 1)
    else:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 1)

class CModel:
  def __init__(self, file_path):
    self.model = self.read_model_from_file(file_path)

  def predict_single(self, image):
    image = cv2.resize(image, (153, 153), interpolation=cv2.INTER_LINEAR)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    p = self.model.predict(input_arr, verbose = 0)
    return round(p[0][0])

  def read_model_from_file(self, file_path):
    json_file = open(file_path, "r")
    model = model_from_json(json_file.read())
    json_file.close()
    model.summary()
    return model
  
  def read_weights_from_file(self, file_path):
    self.model.load_weights(file_path)

def load_model():
    cmodel = CModel("./utils/model0.json")
    cmodel.read_weights_from_file("./utils/weights_model0.h5")
    return cmodel

webApp = Flask(__name__)
webApp.config['UPLOAD_FOLDER'] = './uploads'
webApp.config['DOWNLOAD_FOLDER'] = './downloads'

model = load_model()

detector = get_model("resnet50_2020-07-20", max_size=2048) #value of max_size ?? predition time depends on this
detector.eval()

@webApp.route('/', methods=['GET'])
def test():
    return "Hi, Welcome to Face Mask detector!"

@webApp.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        file_path = os.path.join(webApp.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        parent_image = cv2.imread(file_path)
        faces = retinaFace_detector(file_path, detector)
        print(faces)
        for face in faces:
            print('new face')
            if face != [] and valid_face_coord(face["bbox"]) and face["score"] > 0.95:
                cropped_face = openCV_image_cropping(file_path, face["bbox"])
                p = model.predict_single(cropped_face)
                openCV_draw_boundary_box(parent_image, face['bbox'], p)

        # save predictions to ./downloadds
        file_path_download = os.path.join(webApp.config['DOWNLOAD_FOLDER'], secure_filename(f.filename))
        cv2.imwrite(file_path_download, parent_image)
        return send_file(file_path_download, mimetype='image/jpg')
    return None


if __name__ == "__main__":
    webApp.run(debug = True)