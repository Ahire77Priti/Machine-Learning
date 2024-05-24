import os
import glob
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads/all_class"
STATIC_FOLDER = "static"

# Load model
pretrained_model=InceptionV3(input_shape=(150,150,3),
                             include_top=False,
                             weights='imagenet')
for layers in pretrained_model.layers:
    layers.trainable=False

last_layer=pretrained_model.get_layer('mixed10')
last_output = last_layer.output
x=tf.keras.layers.Flatten()(last_output)

x=tf.keras.layers.Dense(512,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)

x=tf.keras.layers.Dense(256,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)

x=tf.keras.layers.Dense(2,activation='softmax')(x)
model=tf.keras.Model(pretrained_model.input,x)

optimizers = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizers,
              loss='categorical_crossentropy',
               metrics=['accuracy'])

model.load_weights(STATIC_FOLDER + "/best_weights.hdf5")

IMAGE_SIZE = 150

def load_and_preprocess_image():
    test_fldr = 'uploads'
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_fldr,
            target_size = (IMAGE_SIZE, IMAGE_SIZE),
            batch_size = 1,
            class_mode = None,
            shuffle = False)
    test_generator.reset()
    return test_generator


# Predict & classify image
def classify(model):
    batch_size = 1
    test_generator = load_and_preprocess_image()
    prob = model.predict_generator(test_generator, steps=len(test_generator)/batch_size)
    labels_mapping = {0:'Non Oil Spill', 1:'Oil Spill'}
    labels = np.argmax(prob, axis=1)[0]
    label = labels_mapping[labels]
    classified_prob = prob[0][labels]
    return label, classified_prob


# home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/all_class/*.*")
    for filePath in filelist:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model)
        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
    
