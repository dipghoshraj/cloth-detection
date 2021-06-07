import csv
import os
from os.path import isfile, join
from os import listdir
import random

# classes =  ["blouses",   "jackets",  "jumpsuit",  "pallazo",  "skirts",    "sweaters",  "tshirt", "dupattas",  "jeans",    "kurti",     "saree",    "sunglass",  "trousers"]
# headers =  ["image", "blouses",   "jackets",  "jumpsuit",  "pallazo",  "skirts",    "sweaters",  "tshirt", "dupattas",  "jeans",    "kurti",     "saree",    "sunglass",  "trousers"]
classes =  ["Bra", "Handbags",  "Kurtis",  "Shirts",  "Sunglasses",  "Tops",      "Tshirts", "Dupatta",  "Jackets",   "saree",   "Skirts",  "Sweaters",    "Trousers",  "watches"]
headers =  ["image", "Bra", "Handbags",  "Kurtis",  "Shirts",  "Sunglasses",  "Tops",      "Tshirts", "Dupatta",  "Jackets",   "saree",   "Skirts",  "Sweaters",    "Trousers",  "watches"]


def get_image(name):
    dirname = os.path.dirname(__file__)
    dir_n = 'updated_ds/' + name
    data_dir =os.path.join(dirname, dir_n)

    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    data_list = []
    for img in onlyfiles:
        image = os.path.join(dir_n, img)
        data_dict= {
            "image": image,
            "Bra" : 1 if name == 'Bra' else 0,   
            "Handbags": 1 if name == 'Handbags' else 0,  
            "Kurtis": 1 if name == 'Kurtis' else 0,  
            "Shirts": 1 if name == 'Shirts' else 0,  
            "Sunglasses": 1 if name == 'Sunglasses' else 0,   
            "Tops": 1 if name == 'Tops' else 0,  
            "Tshirts": 1 if name == 'Tshirts' else 0, 
            "Dupatta": 1 if name == 'Dupatta' else 0,  
            "Jackets": 1 if name == 'Jackets' else 0,    
            "saree": 1 if name == 'saree' else 0,     
            "Skirts": 1 if name == 'Skirts' else 0,    
            "Sweaters": 1 if name == 'Sweaters' else 0,  
            "Trousers": 1 if name == 'Trousers' else 0,
            "watches": 1 if name == 'watches' else 0
        }

        data_list.append(data_dict)
    return data_list


cloth_tmp = []

for class_name in classes:
    print(class_name)
    cloth_data = get_image(class_name)

    for cloth in cloth_data:
        cloth_tmp.append(cloth)

random.shuffle(cloth_tmp)

filename = "cloths_2.csv"
with open(filename, mode="w", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()

    for c in cloth_tmp:
        writer.writerow(c)