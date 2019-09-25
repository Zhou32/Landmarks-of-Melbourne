# This script is obtained from:  
# https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/
# Author: Adrian Rosebrock
# This script converts from Keras .model file to Core ML .mlmodel file



from keras.models import load_model
import coremltools
# import argparse
import pickle


LABEL_PATH = r"C:\Users\PC-user\Dropbox\workspace-python\Landmarks-of-Melbourne\output\lb-sml"
MODEL_PATH = r"C:\Users\PC-user\Dropbox\workspace-python\Landmarks-of-Melbourne\output\model-sml"
OUTPUT_PATH = r"C:\Users\PC-user\Dropbox\workspace-python\Landmarks-of-Melbourne\output\ml-model-sml"

# USAGE
# python coremlconverter.py --model pokedex.model --labelbin lb.pickle

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained model model")
# ap.add_argument("-l", "--labelbin", required=True,
# 	help="path to label binarizer")
# args = vars(ap.parse_args())


# load the class labels
print("[INFO] loading class labels from label binarizer")
MobileNet_label = pickle.loads(open(LABEL_PATH, "rb").read())
class_labels = MobileNet_label.classes_.tolist()
print("[INFO] class labels: {}".format(class_labels))

# load the trained convolutional neural network
print("[INFO] loading model...")
MobileNet_model = load_model(MODEL_PATH)

# convert the model to coreml format
print("[INFO] converting model")
coreml_model = coremltools.converters.keras.convert(MobileNet_model,
                                                    input_names="image",
                                                    image_input_names="image",
                                                    image_scale=1/255.0,
                                                    class_labels=class_labels,
                                                    is_bgr=True)

# save the model to disk
output = OUTPUT_PATH + ".mlmodel"
print("[INFO] saving model as {}".format(output))
coreml_model.save(output)