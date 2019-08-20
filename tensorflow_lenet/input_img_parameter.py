# ---输入图片设置---begin
# 图片长和宽为28个像素点
img_size = 28
# 图片扁平化处理,大小为28*28=784
img_size_flat = img_size * img_size
# 图片大小
img_shape = (img_size, img_size)
# 值为1,因为是黑白图片
num_img_channels = 1
# 图片最后分类的种类：0~10个数字
num_img_classes = 10
# ---输入图片设置---end