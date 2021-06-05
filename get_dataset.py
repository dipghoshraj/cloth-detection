# import pathlib
# import tensorflow as tf

# from tensorflow import keras

# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)


from os import listdir
import os
from os.path import isfile, join
from typing import Counter

dirname = os.path.dirname(__file__)
data_dir =os.path.join(dirname, 'dataset/jeans')

onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

# print(onlyfiles)
counters = 0
for files in onlyfiles:
    x = 'jeans_' + str(counters) + '.'+ files.split('.')[1]
    source = os.path.join('dataset/jeans', files)
    destination = os.path.join('dataset/jeans', x)

    if files.split('.')[1] == 'jpg':
        print(x, source)
        os.rename(source, destination)
        counters += 1
    else:
        os.remove(source)


# ['blouses', 'dupattas', 'jackets', 'jeans', 'jumpsuits', 'kurti', 'palazzos', 'saree', 'skirts', 'sunglass', 'sweaters', 't-shirt', 'trousers']
