import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


class CarClassifier:
    def __init__(self, model_path, num_classes):
        self.model = tf.keras.models.load_model(model_path)
        self.num_classes = num_classes
        self.class_indices = {}

    def load_classes(self, class_indices):
        self.class_indices = class_indices

    def predict(self, image_path):
        # Загружаем изображение
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Убедимся, что predicted_class содержит целый номер для индексации
        return self.class_indices[predicted_class[0].item()]