from os import listdir
import os
from os.path import isfile, join
from typing import Counter

dirname = os.path.dirname(__file__)
data_dir =os.path.join(dirname, 'dataset/trousers')

onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]


def is_image(filename, verbose=False):
    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True
    
    return False

image_file = []

for files in onlyfiles:
    source = os.path.join('dataset/trousers', files)
    data = is_image(source)

    if not data:
        image_file.append(data)
        os.remove(source)

print(image_file)

