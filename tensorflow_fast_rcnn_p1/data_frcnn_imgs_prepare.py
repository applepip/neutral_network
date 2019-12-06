import os
import random
from shutil import copyfile

classes = ['person', 'mobile_phone', 'car']

for i in range(len(classes)):
    all_imgs = os.listdir(os.path.join("imgs/", classes[i]))
    all_imgs = [f for f in all_imgs if not f.startswith('.')]
    random.seed(1)
    random.shuffle(all_imgs)

    train_imgs = all_imgs[:800]
    test_imgs = all_imgs[800:]

    # Copy each classes' images to train directory
    for j in range(len(train_imgs)):
        original_path = os.path.join(os.path.join('imgs/', classes[i]), train_imgs[j])
        new_path = os.path.join('imgs_train/', train_imgs[j])
        copyfile(original_path, new_path)

    # Copy each classes' images to test directory
    for j in range(len(test_imgs)):
        original_path = os.path.join(os.path.join('imgs/', classes[i]), test_imgs[j])
        new_path = os.path.join('imgs_test/', test_imgs[j])
        copyfile(original_path, new_path)

print('number of training images: ', len(os.listdir('imgs_train/'))) # subtract one because there is one hidden file named '.DS_Store'
print('number of test images: ', len(os.listdir('imgs_test/')))