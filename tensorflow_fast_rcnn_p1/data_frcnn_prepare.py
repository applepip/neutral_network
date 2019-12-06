import pandas as pd
import os
import sys

class_descriptions_fname = 'data/class-descriptions-boxable.csv'

class_descriptions = pd.read_csv(class_descriptions_fname)
print(class_descriptions.head())

person_pd = class_descriptions[class_descriptions['class']=='Person']
phone_pd = class_descriptions[class_descriptions['class']=='Mobile phone']
car_pd = class_descriptions[class_descriptions['class']=='Car']

label_name_person = person_pd['name'].values[0]
label_name_phone = phone_pd['name'].values[0]
label_name_car = car_pd['name'].values[0]

label_names = [label_name_person, label_name_phone, label_name_car]
train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
train_imgs = os.listdir('imgs_train/')
train_imgs = [name for name in train_imgs if not name.startswith('.')]

annotations_bbox = pd.read_csv('data/train-annotations-bbox.csv')
classes = ['Person', 'Mobile phone', 'Car']

for i in range(len(train_imgs)):
    sys.stdout.write('Parse train_imgs ' + str(i) + '; Number of boxes: ' + str(len(train_df)) + '\r')
    sys.stdout.flush()
    img_name = train_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID'] == img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                train_df = train_df.append({'FileName': img_name,
                                            'XMin': row['XMin'],
                                            'XMax': row['XMax'],
                                            'YMin': row['YMin'],
                                            'YMax': row['YMax'],
                                            'ClassName': classes[i]},
                                           ignore_index=True)

test_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
test_imgs = os.listdir('imgs_test/')
test_imgs = [name for name in test_imgs if not name.startswith('.')]

for i in range(len(test_imgs)):
    sys.stdout.write('Parse test_imgs ' + str(i) + '; Number of boxes: ' + str(len(test_df)) + '\r')
    sys.stdout.flush()
    img_name = test_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                test_df = test_df.append({'FileName': img_name,
                                            'XMin': row['XMin'],
                                            'XMax': row['XMax'],
                                            'YMin': row['YMin'],
                                            'YMax': row['YMax'],
                                            'ClassName': classes[i]},
                                           ignore_index=True)

train_df.to_csv(os.path.join('data_pro/', 'train.csv'))
test_df.to_csv(os.path.join('data_pro/', 'test.csv'))