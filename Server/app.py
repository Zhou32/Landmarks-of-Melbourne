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

global graph,   MobileNet_model, MobileNet_label, \
                InceptionV3_model, InceptionV3_label, \
                ResNet50_model, ResNet50_label, \
                Xception_model, Xception_label

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

with graph.as_default():
    # Load model
    MobileNet_model = load_model("./model/model-sml")
    InceptionV3_model = load_model("./model/InceptionV3-dataset15-d299-e15")
    ResNet50_model = load_model("./model/ResNet50-dataset15-d299-e15")
    Xception_model = load_model("./model/Xception-dataset15-d299-e15")

    # Load label encoder
    MobileNet_label = pickle.loads(open("./model/lb-sml", "rb").read())

    # TODO: Add label encoders respectively
    InceptionV3_label = pickle.loads(open("./model/lb-sml", "rb").read())
    ResNet50_label = pickle.loads(open("./model/lb-sml", "rb").read())
    Xception_label = pickle.loads(open("./model/lb-sml", "rb").read())


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

        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        if model_name == "MobileNet_model":
            proba = MobileNet_model.predict(image)[0]
            idx = np.argmax(proba)
            label = MobileNet_label.classes_[idx]
            return label

        if model_name == "InceptionV3_model":
            proba = InceptionV3_model.predict(image)[0]
            idx = np.argmax(proba)
            label = InceptionV3_label.classes_[idx]
            return label

        if model_name == "ResNet50_model":
            proba = ResNet50_model.predict(image)[0]
            idx = np.argmax(proba)
            label = ResNet50_label.classes_[idx]
            return label

        if model_name == "Xception_model":
            proba = Xception_model.predict(image)[0]
            idx = np.argmax(proba)
            label = Xception_label.classes_[idx]
            return label


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
            if model_name in ["MobileNet_model", "InceptionV3_model", "ResNet50_model", "Xception_model"]:
                label = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename), model_name)
                return label
            return "Invalid model name!"
        else:
            return "Invalid image!"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)
