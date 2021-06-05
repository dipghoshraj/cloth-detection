from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os

# load model
dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'flower.model')
filename = os.path.join(dirname, 'cloth.model')
model = load_model(filename)

batch_size = 32
img_height = 180
img_width = 180

img_path= 'img_6.jpeg'

# classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
classes = ['kurti', 'sunglass']

img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# apply gender detection on face
# conf = model.predict(img_array)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(score)

# get label with max accuracy
idx = np.argmax(score)
label = classes[idx]

label = "{}: {:.2f}%".format(label, score[idx] * 100)

print(label)
