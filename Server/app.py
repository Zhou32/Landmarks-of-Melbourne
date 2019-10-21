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

global graph, MobileNet_model, InceptionV3_model, ResNet50_model, Xception_model, landmarks_label_encoder, StreetViewModel, street_label_encoder

graph = tf.get_default_graph()

UPLOAD_FOLDER = './image'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app,
          version='2.0',
          title='Landmarks of Melbourne',
          description='Postera Crescam Laude')
parser = api.parser()
parser.add_argument('file', type=FileStorage, location='files', required=True)

# loading Landmark CSV
landmarks_csv_path = "./description/dataset15-metadata.csv"

landmarks_description = {}
landmarks_coordinate = {}
landmarks_imageUrl = {}

with open(landmarks_csv_path) as landmarks_csv_file:
    csv_content = csv.DictReader(landmarks_csv_file)
    for row in csv_content:
        name = row["Landmark"]
        landmarks_description[name] = row["Description"]
        landmarks_coordinate[name] = row["Coordinate"]
        landmarks_imageUrl[name] = row["Image URL"]

# # loading Street CSV
# street_view_csv_path = "./description/street.csv"
#
# street_description = {}
# street_imageUrl = {}
# street_viewUrl = {}
#
# with open(street_view_csv_path) as street_csv_file:
#     csv_content = csv.DictReader(street_csv_file)
#     for row in csv_content:
#         name = row["Street"]
#         street_description[name] = row["Description"]
#         street_imageUrl[name] = row["Image URL"]
#         street_viewUrl[name] = row["View URL"]


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

    # Load label encoder
    landmarks_label_encoder = pickle.loads(open("./model/label_binarizer-dataset15", "rb").read())


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
def landmark_predict(image_path, model_name):
    with graph.as_default():
        image = cv2.imread(image_path)

        if model_name == "MobileNet":
            image = cv2.resize(image, (224, 224))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = MobileNet_model.predict(image)[0]
            idx = np.argmax(proba)
            label = landmarks_label_encoder.classes_[idx]
            return label, proba

        elif model_name == "InceptionV3":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = InceptionV3_model.predict(image)[0]
            idx = np.argmax(proba)
            label = landmarks_label_encoder.classes_[idx]
            return label, proba

        elif model_name == "ResNet50":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = ResNet50_model.predict(image)[0]
            idx = np.argmax(proba)
            label = landmarks_label_encoder.classes_[idx]
            return label, proba

        elif model_name == "Xception":
            image = cv2.resize(image, (299, 299))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            proba = Xception_model.predict(image)[0]
            idx = np.argmax(proba)
            label = landmarks_label_encoder.classes_[idx]
            return label, proba

        else:
            return "Model name error!"


@print_run_time
def street_predict(image_path):
    with graph.as_default():
        image = cv2.imread(image_path)

        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        proba = StreetViewModel.predict(image)[0]
        idx = np.argmax(proba)
        label = street_label_encoder.classes_[idx]

        return label, proba


@api.route('/landmarks_predict')
class LandmarksPredicate(Resource):
    @api.response(200, "Success to get the image.")
    @api.response(404, "The upload image may be invalid!")
    @api.doc(params={'file': 'upload image'})
    @api.doc(params={'model': 'select model'})
    @api.expect(parser, validate=True)
    def post(self):
        image_name = request.files['file']
        model_name = request.values["model"]

        if image_name:
            label, all_proba = self.process_image(image_name, model_name)
            response = {
                "best_result": label,
                "all_results": all_proba
            }
            return response, 200
        else:
            return "Invalid parameters!", 400

    def process_image(self, image_file, model_name):
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label, proba = landmark_predict(os.path.join(app.config['UPLOAD_FOLDER'], filename), model_name)
            proba = list(proba)

            all_proba = [(label_name, str(proba[index])) for index, label_name in
                         enumerate(landmarks_label_encoder.classes_)]

            return label, all_proba
        else:
            return "Invalid image!", 400


@api.route('/street_predict')
class StreetPredicate(Resource):
    @api.response(200, "Success to get the image.")
    @api.response(404, "The upload image may be invalid!")
    @api.doc(params={'file': 'upload image'})
    @api.expect(parser, validate=True)
    def post(self):
        image_name = request.files['file']

        if image_name:
            label, all_proba = self.process_image(image_name)
            response = {
                "result": label,
                "all_results": all_proba
            }
            return response, 200
        else:
            return "Invalid parameters!", 400

    def process_image(self, image_file):
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            label, proba = street_predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            proba = list(proba)
            all_proba = [(label_name, str(proba[index])) for index, label_name in enumerate(street_label_encoder.classes_)]

            return label, all_proba

        else:
            return "Invalid image!", 400


@api.route('/street_collection')
class StreetPredicate(Resource):
    @api.response(200, "Success to get the image.")
    @api.response(404, "The upload image may be invalid!")
    @api.doc(params={'file': 'upload image'})
    @api.doc(params={'label': 'image label'})
    @api.expect(parser, validate=True)
    def post(self):
        image = request.files['file']
        image_label = request.values["label"]

        if image:
            self.process_image(image, image_label)
            return "Success", 200
        else:
            return "Invalid parameters!", 400

    def process_image(self, image, image_label):
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_label, filename))
        else:
            return "Invalid image!", 400


@api.route('/get_landmark_description')
class GetLandmarksDescription(Resource):
    @api.response(200, "Success!")
    @api.response(404, "The parameter is invaild!")
    @api.doc(params={'Label_name': "Landmark\'s label name"})
    @api.doc(params={'All_result': "Return all description"})
    def post(self):

        all_result = request.values["All_result"]

        if all_result == "False":
            landmark_name = request.values["Label_name"]
            if landmark_name and landmark_name in landmarks_description:
                return {
                           "description": landmarks_description[landmark_name],
                           "coordinate": landmarks_coordinate[landmark_name],
                           "imageUrl": landmarks_imageUrl[landmark_name]
                       }, 200

        elif all_result == "True":
            response = []
            for name in landmarks_description:
                response += [{
                    "name": name,
                    "description": landmarks_description[name],
                    "coordinate": landmarks_coordinate[name],
                    "imageUrl": landmarks_imageUrl[name]
                }]
            return response, 200

        return "Invalid parameters!", 404


@api.route('/get_street_description')
class GetStreetDescription(Resource):
    @api.response(200, "Success!")
    @api.response(404, "The parameter is invaild!")
    @api.doc(params={'Label_name': "Street\'s label name"})
    @api.doc(params={'All_result': "Return all description"})
    def post(self):
        all_result = request.values["All_result"]

        if all_result == "False":
            street_name = request.values["Label_name"]
            if street_name and street_name in street_description:
                return {
                           "description": street_description[street_name],
                           "imageUrl": street_imageUrl[street_name],
                           "streetView": street_viewUrl[street_name]
                       }, 200

        elif all_result == "True":
            response = []
            for name in street_description:
                response += [{
                    "name": name,
                    "description": street_description[name],
                    "imageUrl": street_imageUrl[name],
                    "streetView": street_viewUrl[name]
                }]
            return response, 200

        return "Invalid parameters!", 404


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
