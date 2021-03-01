# Face Mask Detection
Detecting the faces of the people in the scene and classify them as with mask and without mask. <br>
The main task of face mask detection is divided into two subtasks. Face Detection task and Mask Classification task. In the former one, faces of humans from a scene(usually with high background noise) is detected. The detected faces are passed as input to the latter task, where the detected faces with mask and detected faces without mask are classified. More Details are provided in the subsequent sections on the tasks. <br>

# Face Detection Using [RetinaFace](https://arxiv.org/abs/1905.00641) and OpenCV
Pre-trained RetinaFace model is used for face detection task. Open Source Python Package Index implementation available [here](https://pypi.org/project/retinaface-pytorch/) is used.

## Implementations of RetinaFace
The open source official implementation of RetinaFace can be found [here](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace). The next popular implementation which is also available as a Python Package Index  is by Vladimir Iglovikov[@ternaus](https://github.com/ternaus/) which is available [here](https://github.com/ternaus/retinaface). This implementation is build from [biubug6](https://github.com/biubug6/) implmentation, available [here](https://github.com/biubug6/Pytorch_Retinaface). Which is inturn build up on the official implementation.

## Choosing [retinaface-pytorch](https://pypi.org/project/retinaface-pytorch/) over [retinaface](https://pypi.org/project/retinaface/)
Both the Python Package Index implementations are tested on a single [crowd image](https://habrastorage.org/webt/tj/gk/ch/tjgkch5v0x-tubycgzp3pfbrtas.jpeg) with many people in it. retinaface-pytorch implemenation crossed the retinaface implmentation by a huge margin. Only 81 faces were detected in the retinface implementation whereas 416 images were detected in the retinaface-pytorch implementation. But there is also a drawback of time. Both the implementations perform single forward pass through the dataset, but retinaface implementation is a tad faster then retinaface-pytorch. The test results can be found [here](https://colab.research.google.com/drive/1bZPu2y8dAk5yC50PtIERXmvj-fAOtcIX?usp=sharing).

## More About retinaface-pytorch
The model is trained and validated on [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset along with validation in [FDDB](https://drive.google.com/file/d/17t4WULUDgZgiSy5kpCax4aooyPaz3GQH/view). [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) module is used in the implmentation. RetinaFace in this implementation is build on the ResNet50 backbone. The following features are available in the implementation
```
vis_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray # utils.py, to draw faces and landmarks on detected faces
model.eval() # to give out the evaluation result
model.predict_jsons(image: np.array, confidence_threshold: float = 0.7, nms_threshold: float = 0.4) # to make predictions and return list of faces, landmarks and confidence score
get_model(model_name: str, max_size: int, device: str = "cpu") -> Model # returns pre-trained model for usage
```

# Mask Classification Using Deep Learning

# Dataset
## Real World Masked Face Detection(RWMFD) Dataset
[Download here](https://drive.google.com/file/d/1mNZ5eaoT9A0LdXLFZcE4lFeM9X6X7Cjt/view?usp=sharing) <br>
Dataset contains unlabelled images of people with and without mask(typically with heavy background noise). This dataset contains 4194 unlabelled images. This dataset is intended only for testing purposes and not for training. This dataset is extracted from the open source dataset [Real-World-Masked-Face-Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) as part of Open Ended Lab Project at IIT Palakkad. Further details about the dataset can be found in the README file present in the dataset.

## Face Mask ~12K Dataset
[Download here](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) <br>
Dataset contains labelled images of people's faces with masks and without masks. This dataset is intended to train Mask Classification Model. It contains 11784 images, 5883 are masked images and 5901 are unmasked images.

# Performance

# API

# Website

# Contributors
- Veludandi Sai Vipul Mohan [@vipul43](https://github.com/vipul43)
- Pulavarti Vinay Kumar [@pulavartivinay](https://github.com/pulavartivinay)

# Mentor
- Dr Chandra Shekar Lakshminarayanan