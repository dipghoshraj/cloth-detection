import pandas as pd
import requests
import random

filename = 'styles.csv'
images_file = 'images.csv'

data_c = pd.read_csv(filename,  error_bad_lines=False)
images_data = pd.read_csv(images_file,  error_bad_lines=False)

cols = ['id', 'articleType']
data = data_c.query('articleType == "Handbags"')[cols]

id_list = list(data['id'])
random.shuffle(id_list)

id_list = id_list[:200]


print(id_list)

cols = ['link']
images_links = []
for ids in id_list:
    query = "filename == '{{image}}.jpg'"
    query = query.replace("{{image}}", str(ids))
    print(query)
    data = images_data.query(query)[cols]
    link = list(data['link'])
    try:
        filename = 'updated_ds/Handbags/' + str(ids) + '.jpg'
        f = open(filename, 'wb')
        f.write(requests.get(link[0]).content)
        f.close()
    except:
        pass