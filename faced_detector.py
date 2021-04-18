import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import cv2
import numpy as np
import os


# const
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")
YOLO_SIZE = 288
YOLO_TARGET = 9
CORRECTOR_SIZE = 50

# utils
def iou(bbox1, bbox2):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2, bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    boxB = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2, bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    ret = interArea / float(boxAArea + boxBArea - interArea)

    return ret


class FaceDetector(object):

    def __init__(self):
        self.load_model(os.path.join(MODELS_PATH, "face_yolo.pb"))
        self.load_aux_vars()

        self.face_corrector = FaceCorrector()

    def load_aux_vars(self):
        cols = np.zeros(shape=[1, YOLO_TARGET])
        for i in range(1, YOLO_TARGET):
            cols = np.concatenate((cols, np.full((1, YOLO_TARGET), i)), axis=0)

        self.cols = cols
        self.rows = cols.T

    def load_model(self, yolo_model, from_pb=True):
        graph = tf1.Graph()
        with graph.as_default():
            self.sess = tf1.Session()

            if from_pb:
                with tf1.gfile.GFile(yolo_model, "rb") as f:
                    graph_def = tf1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf1.import_graph_def(graph_def, name="") # If not, name is appended in op name

            else:
                ckpt_path = tf1.train.latest_checkpoint(yolo_model)
                saver = tf1.train.import_meta_graph('{}.meta'.format(ckpt_path))
                saver.restore(self.sess, ckpt_path)

            self.img = tf1.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf1.get_default_graph().get_tensor_by_name("training:0")
            self.prob = tf1.get_default_graph().get_tensor_by_name("prob:0")
            self.x_center = tf1.get_default_graph().get_tensor_by_name("x_center:0")
            self.y_center = tf1.get_default_graph().get_tensor_by_name("y_center:0")
            self.w = tf1.get_default_graph().get_tensor_by_name("w:0")
            self.h = tf1.get_default_graph().get_tensor_by_name("h:0")

    # Receives RGB numpy array
    def predict(self, frame, thresh=0.85):
        input_img = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE)) / 255.
        input_img = np.expand_dims(input_img, axis=0)

        pred = self.sess.run([self.prob, self.x_center, self.y_center, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        bboxes = self._absolute_bboxes(pred, frame, thresh)
        bboxes = self._correct(frame, bboxes)
        bboxes = self._nonmax_supression(bboxes)

        return bboxes

    def _absolute_bboxes(self, pred, frame, thresh):
        img_h, img_w, _ = frame.shape
        p, x, y, w, h = pred

        mask = p > thresh

        x += self.cols
        y += self.rows

        p, x, y, w, h = p[mask], x[mask], y[mask], w[mask], h[mask]

        ret = []

        for j in range(x.shape[0]):
            xc, yc = int((x[j]/YOLO_TARGET)*img_w), int((y[j]/YOLO_TARGET)*img_h)
            wi, he = int(w[j]*img_w), int(h[j]*img_h)
            ret.append((xc, yc, wi, he, p[j]))

        return ret

    def _nonmax_supression(self, bboxes, thresh=0.2):
        SUPPRESSED = 1
        NON_SUPPRESSED = 2

        N = len(bboxes)
        status = [None] * N
        for i in range(N):
            if status[i] is not None:
                continue

            curr_max_p = bboxes[i][-1]
            curr_max_index = i

            for j in range(i+1, N):
                if status[j] is not None:
                    continue

                metric = iou(bboxes[i], bboxes[j])
                if metric > thresh:
                    if bboxes[j][-1] > curr_max_p:
                        status[curr_max_index] = SUPPRESSED
                        curr_max_p = bboxes[j][-1]
                        curr_max_index = j
                    else:
                        status[j] = SUPPRESSED

            status[curr_max_index] = NON_SUPPRESSED

        return [bboxes[i] for i in range(N) if status[i] == NON_SUPPRESSED]

    def _correct(self, frame, bboxes):
        N = len(bboxes)
        ret = []

        img_h, img_w, _ = frame.shape
        for i in range(N):
            x, y, w, h, p = bboxes[i]

            MARGIN = 0.5
            # Add margin
            xmin = int(max(0, x - w/2 - MARGIN*w))
            xmax = int(min(img_w, x + w/2 + MARGIN*w))
            ymin = int(max(0, y - h/2 - MARGIN*h))
            ymax = int(min(img_h, y + h/2 + MARGIN*h))

            face = frame[ymin:ymax, xmin:xmax, :]
            x, y, w, h = self.face_corrector.predict(face)

            ret.append((x + xmin, y + ymin, w, h, p))

        return ret


class FaceCorrector(object):

    def __init__(self):
        self.load_model(os.path.join(MODELS_PATH, "face_corrector.pb"))

    def load_model(self, corrector_model, from_pb=True):
        self.graph = tf1.Graph()
        with self.graph.as_default():
            self.sess = tf1.Session()
            if from_pb:
                with tf1.gfile.GFile(corrector_model, "rb") as f:
                    graph_def = tf1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf1.import_graph_def(graph_def, name="") # If not, name is appended in op name

            else:
                ckpt_path = tf1.train.latest_checkpoint(corrector_model)
                saver = tf1.train.import_meta_graph('{}.meta'.format(ckpt_path))
                saver.restore(self.sess, ckpt_path)

            self.img = tf1.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf1.get_default_graph().get_tensor_by_name("training:0")
            self.x = tf1.get_default_graph().get_tensor_by_name("X:0")
            self.y = tf1.get_default_graph().get_tensor_by_name("Y:0")
            self.w = tf1.get_default_graph().get_tensor_by_name("W:0")
            self.h = tf1.get_default_graph().get_tensor_by_name("H:0")

    def predict(self, frame):
        # Preprocess
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (CORRECTOR_SIZE, CORRECTOR_SIZE)) / 255.
        input_img = np.reshape(input_img, [1, CORRECTOR_SIZE, CORRECTOR_SIZE, 3])

        x, y, w, h = self.sess.run([self.x, self.y, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        img_h, img_w, _ = frame.shape

        x = int(x*img_w)
        w = int(w*img_w)

        y = int(y*img_h)
        h = int(h*img_h)

        return x, y, w, h

tf1.enable_v2_behavior()