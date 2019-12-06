import pandas as pd
from skimage import io
import os

subperson_pd = pd.read_csv('data/subperson_img_url.csv')
subphone_pd = pd.read_csv('data/subphone_img_url.csv')
subcar_pd = pd.read_csv('data/subcar_img_url.csv')

subperson_img_url = subperson_pd['image_url'].values
subphone_img_url = subphone_pd['image_url'].values
subcar_img_url = subcar_pd['image_url'].values

# urls = [subperson_img_url, subphone_img_url, subcar_img_url]
#
# classes = ['Person', 'Mobile phone', 'Car']
#
# saved_dirs = ['imgs/person','imgs/mobile_phone','imgs/car']

urls = [subcar_img_url]

classes = ['Car']

saved_dirs = ['imgs/car']

# Download images
for i in range(len(classes)):
    saved_dir = saved_dirs[i]
    t = 695
    for url in urls[i][695:]:
        img = io.imread(url)
        t = t + 1
        print(t, ":", url[-20:])
        saved_path = os.path.join(saved_dir, url[-20:])
        io.imsave(saved_path, img)