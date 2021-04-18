'''
HOW TO RUN THE FLASK APP? FLASK_APP=app.py flask run
HOW TO KILL THE FLASK SERVER? Ctrl + C
HOW TO RUN HOT RELOAD FLASK APP? FLASK_APP=app.py FLASK_ENV=development flask run
'''

import os
import base64
from flask import Flask, request, render_template, jsonify
import cv2
from faced_detector import FaceDetector
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def opencv_image_cropping(img, face):
    '''
    cropping face coordinated from original image for
    sending it down the pipeline to classifier
    '''
    crop_img = img[face[1]:face[3], face[0]:face[2]]
    return crop_img

def valid_face_coord(face):
    '''
    to check if the predicted faces have non-negative coordinates
    '''
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

def opencv_draw_boundary_box(image, face, pred):
    '''
    drawing bounding box in the parent image
    according to the predicted class
    green if the predicted class is 0(with mask)
    and red if the predicted class is 1(without mask)
    '''
    if pred==1:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
    else:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

class CModel:
    '''
    class to contain pre
    '''
    def __init__(self, file_path):
        '''
        reading model from file
        '''
        self.model = self.read_model_from_file(file_path)

    def predict_single(self, image):
        '''
        preprocessing the image for predicting single file
        '''
        image = cv2.resize(image, (153, 153), interpolation=cv2.INTER_LINEAR)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        pred = self.model.predict(input_arr, verbose = 0)
        return round(pred[0][0])

    def read_model_from_file(self, file_path):
        '''
        helper function to read model from file
        '''
        mod = tf.keras.models.load_model(file_path)
        return mod

def load_model(filepath):
    '''
    helper function to load model
    '''
    cmodel = CModel(filepath)
    return cmodel

webApp = Flask(__name__)
webApp.config['UPLOAD_FOLDER'] = './uploads'
webApp.config['DOWNLOAD_FOLDER'] = './downloads'

model = load_model("./utils/model0.h5")

face_detector = FaceDetector()

@webApp.route('/', methods=['GET'])
def test():
    '''
    returns html template to be rendered as home page
    '''
    return render_template('index.html')

@webApp.route('/predict', methods=['POST'])
def predict():
    '''
    
    '''
    if request.method == 'POST':
        file = request.files['image']

        # Save the file to ./uploads
        file_path = os.path.join(webApp.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Make prediction
        parent_image = cv2.imread(file_path)
        rgb_img = cv2.cvtColor(parent_image.copy(), cv2.COLOR_BGR2RGB)

        # Receives RGB numpy image (HxWxC) and
        # returns (x_center, y_center, width, height, prob) tuples. 
        faces = face_detector.predict(rgb_img, 0.5)
        for face in faces:
            if face != [] and valid_face_coord(face):
                x, y, w, h, _ = face
                x = int(x - w/2)
                y = int(y - h/2)
                face_coordinates = [x, y, x+w, y+h]
                cropped_face = opencv_image_cropping(parent_image, face_coordinates)
                pred = model.predict_single(cropped_face)
                opencv_draw_boundary_box(parent_image, face_coordinates, pred)

        # save predictions to ./downloadds
        file_path_download = os.path.join(webApp.config['DOWNLOAD_FOLDER'], secure_filename(file.filename))
        cv2.imwrite(file_path_download, parent_image)

        with open(file_path_download, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return jsonify({'b64image': str(encoded_string)})
    return None

if __name__ == "__main__":
    webApp.run(debug = True)
