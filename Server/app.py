import csv
import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request
from flask_restplus import Api, Resource
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

global graph, MobileNet_model, InceptionV3_model, ResNet50_model, Xception_model, label_encoder

graph = tf.get_default_graph()

UPLOAD_FOLDER = './image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app,
          version='0.1',
          title='Landmarks of Melbourne',
          description='Postera Crescam Laude')
parser = api.parser()
parser.add_argument('file', type=FileStorage, location='files', required=True)

# loading CSV
csv_path = "./description/dataset15-metadata.csv"

landmarks_description = {}
landmarks_coordinate = {}
landmarks_imageUrl = {}

with open(csv_path) as csv_file:
    csv_content = csv.DictReader(csv_file)
    for row in csv_content:
        name = row["Landmark"]
        landmarks_description[name] = row["Description"]
        landmarks_coordinate[name] = row["Coordinate"]
        landmarks_imageUrl[name] = row["Image URL"]

with graph.as_default():
    # Load model
    print("Loading MobileNet_model...")
    MobileNet_model = load_model("./model/MobileNetV2-dataset15-d224-e15")

    print("Loading InceptionV3_model...")
    InceptionV3_model = load_model("./model/InceptionV3-dataset15-d299-e15")

    print("Loading ResNet50_model...")
    ResNet50_model = load_model("./model/ResNet50-dataset15-d299-e15")

    print("Loading Xception_model...")
    Xception_model = load_model("./model/Xception-dataset15-d299-e15")
    #
    # Load label encoder
    label_encoder = pickle.loads(open("./model/label_binarizer-dataset15", "rb").read())


# Timer
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        result = func(*args, **kw)
        print('current Function [%s] run time is %.2f ms' % (func.__name__, (time.time() - local_time) * 1000))
        return result

    return wrapper


# Todo: Moving to Utils
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@print_run_time
def predict(image_path, model_name):
    with graph.as_default():
        image = cv2.imread(image_path)

        if model_name == "MobileNet":
            image = cv2.resize(image, (224, 224))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = MobileNet_model.predict(image)[0]
            idx = np.argmax(proba)
            label = label_encoder.classes_[idx]
            return label

        elif model_name == "InceptionV3":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = InceptionV3_model.predict(image)[0]
            idx = np.argmax(proba)
            label = label_encoder.classes_[idx]
            return label

        elif model_name == "ResNet50":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = ResNet50_model.predict(image)[0]
            idx = np.argmax(proba)
            label = label_encoder.classes_[idx]
            return label

        elif model_name == "Xception":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = Xception_model.predict(image)[0]
            idx = np.argmax(proba)
            label = label_encoder.classes_[idx]
            return label

        else:
            return "Model name error!"


@api.route('/upload_image')
class ImagePredicate(Resource):
    @api.response(200, "Success to get the image.")
    @api.response(404, "The upload image may be invalid!")
    @api.doc(params={'file': 'upload image'})
    @api.doc(params={'model': 'select model'})
    @api.expect(parser, validate=True)
    def post(self):
        image_name = request.files['file']
        model_name = request.values["model"]

        if image_name:
            return self.process_image(image_name, model_name), 200
        else:
            return "Invalid parameters!", 404

    def process_image(self, image_file, model_name):
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename), model_name)

            return label
        else:
            return "Invalid image!"


@api.route('/get_description')
class GetDescription(Resource):
    @api.response(200, "Success!")
    @api.response(404, "The parameter is invaild!")
    @api.doc(params={'Label_name': "Landmark\'s label name"})
    def post(self):
        landmark_name = request.values["Label_name"]

        if landmark_name in landmarks_description:
            return {
                    "description":landmarks_description[landmark_name],
                    "coordinate":landmarks_coordinate[landmark_name],
                    "imageUrl":landmarks_imageUrl[landmark_name]
                    }
        else:
            return "Invalid parameters!", 404


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)

