from struct import unpack
import os

from os import listdir
from os import listdir
import os
from os.path import isfile, join



marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        




def get_bads(name):
    bads = []
    dirname = os.path.dirname(__file__)
    dir_n = 'small_dataset/' + name
    data_dir =os.path.join(dirname, dir_n)

    onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

    for img in onlyfiles:
        image = os.path.join(dir_n, img)
        image = JPEG(image) 
        try:
            image.decode()   
        except:
            bads.append(os.path.join(dir_n, img))
    return bads


classes =  ["blouses",   "jackets",  "jumpsuit",  "pallazo",  "skirts",    "sweaters",  "tshirt", "dupattas",  "jeans",    "kurti",     "saree",    "sunglass",  "trousers"]


for class_name in classes:
    print(class_name)
    bads = get_bads(class_name)
    print(bads)
    for source in bads:
        os.remove(source)
#   os.remove(osp.join(root_img,name))

