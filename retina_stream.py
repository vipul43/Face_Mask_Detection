'''
HOW TO RUN THE FLASK APP? FLASK_APP=app.py flask run
HOW TO KILL THE FLASK SERVER? Ctrl + C
HOW TO RUN HOT RELOAD FLASK APP? FLASK_APP=app.py FLASK_ENV=development flask run
'''

import os
import random
import cv2 
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import tensorflow as tf
import numpy as np

webcam = cv2.VideoCapture(0)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def retinaFace_detector(image, detector):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #detector accepts only RGB images
    faces = detector.predict_jsons(image)
    return faces

def openCV_image_cropping(img, face):
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
    if p==0:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
    else:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

class CModel:
  def __init__(self, file_path):
    self.model = self.read_model_from_file(file_path)

  def predict_single(self, image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    p = self.model.predict(input_arr, verbose = 0)[0]
    print(p)
    return np.argmax(p)

  def read_model_from_file(self, file_path):
    model = tf.keras.models.load_model(file_path)
    model.summary()
    return model

def load_model():
    cmodel = CModel("./utils/modelV3_inception_resnet_v2_35epochs_AIZOO.h5")
    return cmodel

model = load_model()

detector = get_model("resnet50_2020-07-20", max_size=2048) #value of max_size ?? predition time depends on this
detector.eval()

skip_frame = 0
while True:
    is_frame_read_success, frame = webcam.read()
    print(frame.shape)
    frame_height, frame_width = frame.shape[:2]
    if is_frame_read_success:
        # Make prediction
        faces = retinaFace_detector(frame, detector)
        for face in faces:
            if face != [] and valid_face_coord(face["bbox"]) and face["score"] > 0.95:
                cropped_face = openCV_image_cropping(frame, face["bbox"])
                p = model.predict_single(cropped_face)
                openCV_draw_boundary_box(frame, face['bbox'], p)
        cv2.imshow('FACE MASK DETECTOR', frame)
        key = cv2.waitKey(1)
    else:
        print("Failed to Load")
        break

    if(key==81 or key==113):
        break

webcam.release()