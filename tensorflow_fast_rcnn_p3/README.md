1. 执行tensorflow_fast_rcnn_p1中data_img_download.py，下载图片数据到imgs文件夹
2. 执行tensorflow_fast_rcnn_p1中data_frcnn_imgs_prepare.py，将图片分成imgs_test和imgs_train两个图片文件夹
3. 执行tensorflow_fast_rcnn_p1中data_frcnn_prepare.py生成train.csv和test.csv，内容包括图片path、Ground Truth位置、类名，
即['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName']，
4. 执行data_test_annotation.py和data_train_annotation.py生成内容包括图片path、Ground Truth相对像素位置、类名，
即['FileName', 'YMin','XMin', 'XMax',  'YMax', 'ClassName']
5.然后可以在tensorflow_fast_rcnn_p2和tensorflow_fast_rcnn_p4中训练模型
6.tensorflow_fast_rcnn_p3中测试训练好的模型。

注：

tensorflow_fast_rcnn_p2是cpu计算，tensorflow_fast_rcnn_p4是gpu计算。

在训练时需要先下载基础模型vgg16_weights_tf_dim_ordering_tf_kernels.h5到model文件夹里。
