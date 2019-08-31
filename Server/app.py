from keras.preprocessing.image import img_to_array
from keras.models import Sequential, Model, load_model
import numpy as np
import pickle
import cv2
import os
from flask import Flask, request
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import tensorflow as tf
import time


global graph, model, lb
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
    model = load_model("./model/model-sml")
    # Load label encoder
    lb = pickle.loads(open("./model/lb-sml", "rb").read())


# Timer
def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        result = func(*args, **kw)
        print('current Function [%s] run time is %.2f ms' % (func.__name__, (time.time() - local_time)*1000))
        return result
    return wrapper


# Todo: Moving to Utils
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@print_run_time
def predict(image_path):
    with graph.as_default():
        image = cv2.imread(image_path)

        image = cv2.resize(image, (224, 224))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]

    return label


@api.route('/upload_image')
class ImagePredicate(Resource):
    @api.response(200, "Success to get the image.")
    @api.response(404, "The upload image may be invalid!")
    @api.doc(params={'file': 'the upload image'})
    @api.expect(parser, validate=True)
    def post(self):
        image_name = request.files['file']

        if image_name:
            return self.process_image(image_name), 200
        else:
            return "Invalid image!", 404

    def process_image(self, image_file):
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            label = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return label
        else:
            return "Invalid image!"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)


