_E='/classify'
_D='POST'
_C='DOWNLOAD_FOLDER'
_B='UPLOAD_FOLDER'
_A=False
from flask import Flask,render_template,request,jsonify
from flask_socketio import SocketIO,emit
from retinaface import RetinaFace
from werkzeug.utils import secure_filename
import cv2,os,numpy as np,tensorflow as tf,base64
from PIL import Image
import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'
webApp=Flask(__name__)
webApp.config[_B]='./uploads'
webApp.config[_C]='./downloads'
socketio=SocketIO(webApp)
cfg='./utils/tiny-yolo-widerface.cfg'
weights='./utils/tiny-yolo-widerface_final.weights'
net=cv2.dnn.readNet(weights,cfg)
def get_outputs_names(net):A=net.getLayerNames();return[A[B[0]-1]for B in net.getUnconnectedOutLayers()]
def valid_face_coord(face):
	A=face
	if len(A)>=4:
		if A[0]<0:return _A
		if A[1]<0:return _A
		if A[2]<0:return _A
		if A[3]<0:return _A
	return True
def openCV_draw_boundary_box(image,face,p):
	B=image;A=face
	if p==0:cv2.rectangle(B,(A[0],A[1]),(A[2],A[3]),(255,0,0),2)
	else:cv2.rectangle(B,(A[0],A[1]),(A[2],A[3]),(0,255,0),2)
def openCV_image_cropping(img,face):A=face;B=img[A[1]:A[3],A[0]:A[2]];return B
class CModel:
	def __init__(A,file_path):A.model=A.read_model_from_file(file_path)
	def predict_single(C,image):A=image;A=cv2.resize(A,(224,224),interpolation=cv2.INTER_NEAREST);B=tf.keras.preprocessing.image.img_to_array(A);B=np.array([B]);D=C.model.predict(B,verbose=0)[0];return np.argmax(D)
	def read_model_from_file(B,file_path):A=tf.keras.models.load_model(file_path);return A
image_model=CModel('./utils/facemask12k_100epochs_xception_224_rescaled.h5')
stream_model=CModel('./utils/AIZOO_100epochs_mobilenetv2_224_rescaled.h5')
def magic1(frame,file_path):
	D='facial_area';B=frame;C=RetinaFace.detect_faces(file_path)
	for E in C:
		A=C[E]
		if A!=[]and valid_face_coord(A[D])and A['score']>0.95:F=openCV_image_cropping(B,A[D]);G=image_model.predict_single(F);openCV_draw_boundary_box(B,A[D],G)
	return B
def magic2(frame):
	C=frame;G,H=C.shape[:2];J=cv2.dnn.blobFromImage(C,0.00392,(416,416),(0,0,0),True,crop=_A);net.setInput(J);K=net.forward(get_outputs_names(net))
	for L in K:
		for D in L:
			I=D[5:];M=np.argmax(I);N=I[M]
			if N>0.5:
				O=int(D[0]*H);P=int(D[1]*G);E=int(D[2]*H);F=int(D[3]*G);A=int(O-E/2);B=int(P-F/2);Q=[A,B,A+E,B+F]
				if valid_face_coord(Q):R=openCV_image_cropping(C,[A,B,A+E,B+F]);S=stream_model.predict_single(R);openCV_draw_boundary_box(C,[A,B,A+E,B+F],S)
	return C
@webApp.route('/',methods=['GET'])
def index():return render_template('index.html')
@webApp.route('/predict',methods=[_D])
def predict():
	if request.method==_D:
		D=request.files['image'];E='temp.'+D.filename.split('.')[1];A=os.path.join(webApp.config[_B],secure_filename(E));D.save(A);B=Image.open(A);C=np.array(B);C=magic1(C,A);B=Image.fromarray(C);F=os.path.join(webApp.config[_C],secure_filename(E));B.save(F)
		with open(F,'rb')as G:H=base64.b64encode(G.read())
		return jsonify({'b64image':str(H)})
	return None
@socketio.on('input image',namespace=_E)
def classify(input):A=input.split(',')[1];A=base64.b64decode(A);B=Image.open(io.BytesIO(A));C=np.array(B);C=magic2(C);B=Image.fromarray(C);D=io.BytesIO();B.save(D,format='JPEG');A=base64.b64encode(D.getvalue());E=A.decode('utf-8');F='data:image/jpeg;base64,'+str(E);emit('out-image-event',{'image_data':F},namespace=_E)
if __name__=='__main__':socketio.run(webApp)