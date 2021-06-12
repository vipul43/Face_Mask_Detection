'''
HOW TO RUN THE FLASK APP? FLASK_APP=app.py flask run
HOW TO KILL THE FLASK SERVER? Ctrl + C
HOW TO RUN HOT RELOAD FLASK APP? FLASK_APP=app.py FLASK_ENV=development flask run
'''

from flask import Flask, request, render_template, Response
import cv2
import os
import numpy as np
import tensorflow as tf
from imutils.video import FPS
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

webApp = Flask(__name__)

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
    if p==0:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)
    else:
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

def openCV_image_cropping(img, face):
    crop_img = img[face[1]:face[3], face[0]:face[2]]
    return crop_img

class CModel:
  def __init__(self, file_path):
    self.model = self.read_model_from_file(file_path)

  def predict_single(self, image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    p = self.model.predict(input_arr, verbose = 0)[0]
    return np.argmax(p)

  def read_model_from_file(self, file_path):
    model = tf.keras.models.load_model(file_path)
    model.summary()
    return model

model = CModel("./utils/modelV3_mobile_net_v2_10epochs_AIZOO_rescaling.h5")

def gen_frames(camera):
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # dl on frame
            frame_height, frame_width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_outputs_names(net))
            for out in outs:
                for face in out:
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
                    
                        face_co_ordinates = [x, y, x+w, y+h]
                        if(valid_face_coord(face_co_ordinates)):
                            cropped_face = openCV_image_cropping(frame, [x, y, x+w, y+h])
                            p = model.predict_single(cropped_face)
                            openCV_draw_boundary_box(frame, [x, y, x+w, y+h], p)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@webApp.route('/', methods=['GET'])
def test():
    return render_template('index.html')

@webApp.route('/video_feed', methods=['GET'])
def video_feed():
    camera = cv2.VideoCapture(0)
    # without fps FRAME RATE: 1.082726828253353
    # without fps FRAME RATE: 1.1389898701336085
    fps = FPS().start()
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    webApp.run(debug = True)