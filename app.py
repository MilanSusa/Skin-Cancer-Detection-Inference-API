import os

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


MODEL = load_model('./pretrained_models/mobilenet.h5',
                   custom_objects={
                       'top_2_accuracy': top_2_accuracy,
                       'top_3_accuracy': top_3_accuracy
                   })


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/v1/inference', methods=['POST'])
def perform_inference():
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        test_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_data_gen = test_img_gen.flow_from_directory(directory='static',
                                                         target_size=(224, 224),
                                                         color_mode='rgb')

        preds = MODEL.predict_generator(generator=test_data_gen,
                                        steps=1)

    return jsonify({
        "akiec": str(preds[0][0]),
        "bcc": str(preds[0][1]),
        "bkl": str(preds[0][2]),
        "df": str(preds[0][3]),
        "mel": str(preds[0][4]),
        "nv": str(preds[0][5]),
        "vasc": str(preds[0][6])
    })


if __name__ == '__main__':
    app.run()
