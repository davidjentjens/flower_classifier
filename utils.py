import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

def get_params():
  # Initialize parser
  parser = argparse.ArgumentParser()
  
  # Adding non-optional arguments
  parser.add_argument('image_path', action="store", type=str)
  parser.add_argument('model_file', action="store", type=str)

  # Adding optional arguments
  parser.add_argument("-t", "--top_k", help = "Return the top K most likely classes")
  parser.add_argument("-c", "--category_names", help = "Path to a JSON file mapping labels to flower names")

  # Read arguments from command line
  args_dict = vars(parser.parse_args())

  image_path = args_dict['image_path']
  model_file = args_dict['model_file']

  top_k = int(args_dict['top_k'])
  if(top_k == None):
    top_k = 5

  category_names_path = args_dict['category_names']
  if(category_names_path == None):
    category_names_path = 'label_map.json'

  return image_path, model_file, top_k, category_names_path


def process_image(image, image_size):
  tf_image = tf.convert_to_tensor(image)
  tf_image = tf.image.resize(image, (image_size, image_size))
  tf_image /= 255
  image = tf_image.numpy()
  return image

def predict(image_path, loaded_model, top_k):
  image_raw = Image.open(image_path)
  image = np.asarray(image_raw)
  image = process_image(image, 224)
  image = np.expand_dims(image, axis=0)
  
  ps = loaded_model.predict(image)
  
  ind = np.argpartition(ps[0], -top_k)[-top_k:]
  class_indices = ind.tolist()
  
  probs = [ps[0][i] for i in class_indices]
  classes = [str(i+1) for i in class_indices]
      
  return probs, classes