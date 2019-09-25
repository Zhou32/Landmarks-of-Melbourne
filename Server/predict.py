from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet169
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
# from SmallerVGGNet.smallervggnet import SmallerVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras import backend as K
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import pickle
import cv2
import os
import tensorflow as tf
global graph,MobileNet_model
graph = tf.get_default_graph()

# Load model
MobileNet_model = load_model("model/model-sml")
# Load label encoder
label_encoder = pickle.loads(open("model/lb-sml", "rb").read())


def predict(image_path):
    print(image_path)
    image = cv2.imread(image_path)

    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    with graph.as_default():
        proba = MobileNet_model.predict(image)[0]
    idx = np.argmax(proba)
    label = label_encoder.classes_[idx]
    
    return label
