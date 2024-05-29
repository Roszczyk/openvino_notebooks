import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
from pathlib import Path

os.chdir(Path(__file__).parent)

img_width, img_height = 150, 150
model_path = 'model/classification_model.h5'
if not os.path.exists(model_path):
    print(f"Plik {model_path} nie istnieje.")
else:
    print("ISTNIEJE")
model_weights_path = 'model/classification_weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    #answer = np.argmax(result)
    print(result)

counter = 1
while counter<=2550:
    #name ='cat2.jpg'
    name = 'test/.'+str(counter)+'.jpg'
    result = predict(name)
    counter = counter +1