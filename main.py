import keras
from matplotlib.image import imread
import numpy as np
import json
import cv2
from keras.preprocessing import image



path_to_model_folder = 'path/to/model_fodler/'
path_to_img = '/path/to/img.jpg'

if __name__ == '__main__':
    loaded_model= keras.models.load_model(path_to_model_folder)
    img = image.load_img(path_to_img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    final_image = keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    pred = loaded_model.predict(final_image)

    with open('class_indices_file.json', 'r') as f:
        data = json.load(f)
    prediction_name='Failed'
    for k,v in data.items():
        if (v == pred.argmax(axis=1)):
            prediction_name = k
    print('predicted species: '+ str(prediction_name))





