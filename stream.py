import cv2
import tensorflow as tf
import numpy as np

webcam = cv2.VideoCapture(0)
# trained_face_data = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')

cfg = "./utils/yolov3-face.cfg"
weights = "./utils/yolov3-wider_16000.weights"
net = cv2.dnn.readNetFromDarknet(cfg, weights)
print(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def openCV_draw_boundary_box(image, face, p):
    if p==1:
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
    image = cv2.resize(image, (153, 153), interpolation=cv2.INTER_LINEAR)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    p = self.model.predict(input_arr, verbose = 0)
    return round(p[0][0])

  def read_model_from_file(self, file_path):
    model = tf.keras.models.load_model(file_path)
    model.summary()
    return model

model = CModel("./utils/model0.h5")

skip_frame = 0
while True:
    is_frame_read_success, frame = webcam.read()
    if is_frame_read_success:
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(gray_frame.shape)
        blob = cv2.dnn.blobFromImage(frame, 1/255, (480, 640),[0, 0, 0], 1, crop=False)
        print(blob.shape)
        net.setInput(blob)
        print(net)
        print(net.getLayerNames())
        print(net.getUnconnectedOutLayers())
        # faces = net.forward([net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()])
        faces = net.forward()
        print(faces.shape)
        print(faces)
        for face in faces:
            (x, y, w, h)=face
            cropped_face = openCV_image_cropping(frame, [x, y, x+w, y+h])
            p = model.predict_single(cropped_face)
            openCV_draw_boundary_box(frame, [x, y, x+w, y+h], p)
        cv2.imshow('FACE MASK DETECTOR', frame)
        key = cv2.waitKey(1)
    else:
        print("Failed to Load")
        break

    if(key==81 or key==113):
        break

webcam.release()