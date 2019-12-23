1. 执行data_img_download.py，下载图片数据到imgs文件夹
2. 执行data_frcnn_imgs_prepare.py，将图片分成imgs_test和imgs_train两个图片文件夹
3. 执行data_frcnn_prepare.py生成train.csv和test.csv，内容包括图片path、Ground Truth位置、类名，
即['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName']，
4. 执行data_test_annotation.py和data_train_annotation.py生成内容包括图片path、Ground Truth相对像素位置、类名，
即['FileName', 'YMin','XMin', 'XMax',  'YMax', 'ClassName']
