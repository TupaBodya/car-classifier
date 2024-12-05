from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
from model import CarClassifier

app = Flask(__name__)

# Настройка папки для загрузки
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Загрузка модели
model_path = 'car_classifier_model.h5'  # Укажите путь к вашей модели
car_classifier = CarClassifier(model_path, num_classes=3)  # Замените на количество классов

# Загрузка классов
class_indices = {0: 'Minivan', 1: 'Offroad', 2: 'Sedan'}  # Замените на ваши классы
car_classifier.load_classes(class_indices)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Предсказание
        predicted_label = car_classifier.predict(file_path)

        return render_template('result.html', filename=filename, predicted_label=predicted_label)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file provided'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Предсказание
        predicted_label = car_classifier.predict(file_path)

        return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)