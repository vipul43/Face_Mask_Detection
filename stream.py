import cv2
import tensorflow as tf
import numpy as np

webcam = cv2.VideoCapture(0)
# trained_face_data = cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')

cfg = "./utils/yolov2-face.cfg"
weights = "./utils/yolov2.weights"
net = cv2.dnn.readNet(weights, cfg)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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
    print(frame.shape)
    frame_height, frame_width = frame.shape[:2]
    if is_frame_read_success:
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
        cv2.imshow('FACE MASK DETECTOR', frame)
        key = cv2.waitKey(1)
    else:
        print("Failed to Load")
        break

    if(key==81 or key==113):
        break

webcam.release()