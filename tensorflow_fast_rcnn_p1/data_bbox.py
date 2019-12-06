import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
import cv2

mages_boxable_fname = 'data/train-images-boxable.csv'
annotations_bbox_fname = 'data/train-annotations-bbox.csv'
class_descriptions_fname = 'data/class-descriptions-boxable.csv'

images_boxable = pd.read_csv(mages_boxable_fname)

annotations_bbox = pd.read_csv(annotations_bbox_fname)

class_descriptions = pd.read_csv(class_descriptions_fname)

img_name = images_boxable['image_name'][0]
img_url = images_boxable['image_url'][0]

img = io.imread(img_url)

height, width, _ = img.shape
print(img.shape)

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(img)

img_id = img_name[:16]
bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
img_bbox = img.copy()

for index, row in bboxs.iterrows():
    xmin = row['XMin']
    xmax = row['XMax']
    ymin = row['YMin']
    ymax = row['YMax']
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)

    label_name = row['LabelName']
    class_series = class_descriptions[class_descriptions['name'] == label_name]

    class_name = class_series['class'].values[0]
    cv2.rectangle(img_bbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bbox, class_name, (xmin, ymin - 10), font, 1, (0, 255, 0), 2)

plt.subplot(1, 2, 2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()