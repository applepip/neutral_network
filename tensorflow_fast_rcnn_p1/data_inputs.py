import pandas as pd
import numpy as np
from skimage import io

mages_boxable_fname = 'data/train-images-boxable.csv'
annotations_bbox_fname = 'data/train-annotations-bbox.csv'
class_descriptions_fname = 'data/class-descriptions-boxable.csv'

images_boxable = pd.read_csv(mages_boxable_fname)
print(images_boxable.head())

annotations_bbox = pd.read_csv(annotations_bbox_fname)
print(annotations_bbox.head())

class_descriptions = pd.read_csv(class_descriptions_fname)
print(class_descriptions.head())

print('length of the images_boxable: %d' %(len(images_boxable)) )
print('First image in images_boxable👇')

img_name = images_boxable['image_name'][0]
img_url = images_boxable['image_url'][0]

print('\t image_name: %s' % (img_name))
print('\t img_url: %s' % (img_url))

print('')
print('length of the annotations_bbox: %d' %(len(annotations_bbox)))
print('The number of bounding boxes are larger than number of images.')
print('')
print('length of the class_descriptions: %d' % (len(class_descriptions)-1))
# img = io.imread(img_url)
# # io.imshow(img)
# # io.show()

# Get subset of the whole dataset
# Find the label_name for 'Person', 'Mobile Phone' and 'Car' classes
person_pd = class_descriptions[class_descriptions['class']=='Person']
phone_pd = class_descriptions[class_descriptions['class']=='Mobile phone']
car_pd = class_descriptions[class_descriptions['class']=='Car']

label_name_person = person_pd['name'].values[0]
label_name_phone = phone_pd['name'].values[0]
label_name_car = car_pd['name'].values[0]

print(person_pd)
print(phone_pd)
print(car_pd)

person_bbox = annotations_bbox[annotations_bbox['LabelName']==label_name_person]
phone_bbox = annotations_bbox[annotations_bbox['LabelName']==label_name_phone]
car_bbox = annotations_bbox[annotations_bbox['LabelName']==label_name_car]

print('There are %d persons in the dataset' %(len(person_bbox)))
print('There are %d phones in the dataset' %(len(phone_bbox)))
print('There are %d cars in the dataset' %(len(car_bbox)))

person_img_id = person_bbox['ImageID']
phone_img_id = phone_bbox['ImageID']
car_img_id = car_bbox['ImageID']

person_img_id = np.unique(person_img_id)
phone_img_id = np.unique(phone_img_id)
car_img_id = np.unique(car_img_id)
print('There are %d images which contain persons' % (len(person_img_id)))
print('There are %d images which contain phones' % (len(phone_img_id)))
print('There are %d images which contain cars' % (len(car_img_id)))

import random
# Shuffle the ids and pick the first 1000 ids
copy_person_id = person_img_id.copy()
random.seed(1)
random.shuffle(copy_person_id)

copy_phone_id = phone_img_id.copy()
random.seed(1)
random.shuffle(copy_phone_id)

copy_car_id = car_img_id.copy()
random.seed(1)
random.shuffle(copy_car_id)

n = 1000
subperson_img_id = copy_person_id[:n]
subphone_img_id = copy_phone_id[:n]
subcar_img_id = copy_car_id[:n]

print(subperson_img_id[10])
print(subphone_img_id[10])
print(subcar_img_id[10])

# This might takes a while to search all these urls
subperson_img_url = [images_boxable[images_boxable['image_name']==name+'.jpg'] for name in subperson_img_id]
subphone_img_url = [images_boxable[images_boxable['image_name']==name+'.jpg'] for name in subphone_img_id]
subcar_img_url = [images_boxable[images_boxable['image_name']==name+'.jpg'] for name in subcar_img_id]

subperson_pd = pd.DataFrame()
subphone_pd = pd.DataFrame()
subcar_pd = pd.DataFrame()

for i in range(len(subperson_img_url)):
    subperson_pd = subperson_pd.append(subperson_img_url[i], ignore_index=True)
    subphone_pd = subphone_pd.append(subphone_img_url[i], ignore_index=True)
    subcar_pd = subcar_pd.append(subcar_img_url[i], ignore_index=True)

subperson_pd.to_csv('data/subperson_img_url.csv')
subphone_pd.to_csv('data/subphone_img_url.csv')
subcar_pd.to_csv('data/subcar_img_url.csv')