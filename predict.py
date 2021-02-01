import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from utils import get_params, predict

image_path, model_file, top_k, category_names_path = get_params()

with open(category_names_path, 'r') as f:
  class_names = json.load(f)

loaded_model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})

probs, classes = predict(image_path, loaded_model, top_k)

top_class_names = [class_names[f'{i}'] for i in classes]

print(top_class_names)