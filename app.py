import os
import shutil

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

UPLOAD_FOLDER = 'skin_cancer_detection_inference_api/static/images'
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


def remove_image():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_rel_dir = 'skin_cancer_detection_inference_api' + os.path.sep + 'static' + os.path.sep + 'images' + os.path.sep
    folder_abs_dir = os.path.join(base_dir, folder_rel_dir)

    for filename in os.listdir(folder_abs_dir):
        file_path = os.path.join(folder_abs_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}: {e}')


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
        remove_image()
        data = {
            "akiec": str(preds[0][0]),
            "bcc": str(preds[0][1]),
            "bkl": str(preds[0][2]),
            "df": str(preds[0][3]),
            "mel": str(preds[0][4]),
            "nv": str(preds[0][5]),
            "vasc": str(preds[0][6])
        }

        return jsonify({"data": data})


if __name__ == '__main__':
    app.run()
