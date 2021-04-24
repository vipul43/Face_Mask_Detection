import cv2
import tensorflow as tf
import numpy as np
from faced_detector import FaceDetector

face_detector = FaceDetector()

webcam = cv2.VideoCapture(0)
# trained_face_data = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')

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
    if pred==0:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
    else:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

def opencv_image_cropping(img, face):
    '''
    cropping face coordinated from original image for
    sending it down the pipeline to classifier
    '''
    crop_img = img[face[1]:face[3], face[0]:face[2]]
    return crop_img

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
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        pred = self.model.predict(input_arr, verbose = 0)[0]
        print(pred)
        return np.argmax(pred)

    def read_model_from_file(self, file_path):
        '''
        helper function to read model from file
        '''
        mod = tf.keras.models.load_model(file_path)
        return mod

model = CModel("./utils/modelV3_inception_resnet_v2_35epochs_AIZOO.h5")

skip_frame = 0
while True:
    is_frame_read_success, frame = webcam.read()
    print(frame.shape)
    if is_frame_read_success:
        print("success")
        rgb_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        faces = face_detector.predict(rgb_img, 0.5)
        for face in faces:
            if face != [] and valid_face_coord(face):
                x, y, w, h, _ = face
                x = int(x - w/2)
                y = int(y - h/2)
                face_coordinates = [x, y, x+w, y+h]
                cropped_face = opencv_image_cropping(frame, face_coordinates)
                pred = model.predict_single(cropped_face)
                opencv_draw_boundary_box(frame, face_coordinates, pred)
        cv2.imshow('FACE MASK DETECTOR', frame)
        key = cv2.waitKey(1)
    else:
        print("Failed to Load")
        break

    if(key==81 or key==113):
        break

webcam.release()