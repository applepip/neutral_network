import pandas as pd
import os
import cv2

# print(train_df.head())
train_df = pd.read_csv('data_pro/train.csv')

f = open("annotation/train_annotation.txt", "w+")
for idx, row in train_df.iterrows():
    img = cv2.imread(('imgs_train/' + row['FileName']))
    height, width = img.shape[:2]
    x1 = int(row['XMin'] * width)
    x2 = int(row['XMax'] * width)
    y1 = int(row['YMin'] * height)
    y2 = int(row['YMax'] * height)

    fileName = os.path.join('imgs_train/', row['FileName'])
    className = row['ClassName']
    f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')
f.close()