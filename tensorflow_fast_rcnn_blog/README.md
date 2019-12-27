# 基于R-CNNs的目标检测和分类
------

**作者：Ankur Mohan    译者：leon**

R-CNN对于一般图片集进行检测和识别是非常高效的，它的mAP得分同等高于以前的技术，Ross Girshick 和 al将R-CNN的方法在下面的论文中进行了描述。

1. R-CNN (Girshick et al. 2013)*
2. Fast R-CNN (Girshick 2015)*
3. Faster R-CNN (Ren et al. 2015)*

最新的R-CNN方法是在最近的一篇论文Faster R-CNN中进行描述的，我最先考虑重第一篇论文到最后一篇论文描述该方法的演变，但我发现那是一个充满野心的事业。最后我静下来开始详细在博客中介绍最后一篇论文中的Faster R-CNN方法。

非常幸运，这里有很多实现R-CNN算法的工具，比如TensorFlow, PyTorch和其他机器学习的库。我的实现方法在已上传到我的[github](https://github.com/ruotianluo/pytorch-faster-rcnn)上。

在此文种使用的很多术语（比如，不同layer的名字）与代码中的保持一致。理解该文的信息将使你更容易的理解PyTorch和你自己进行实践。

## 文章组成

**section 1 - 图片的预处理**：本节中，我们将描述输入图片的预处理过程。这些过程包括平均像素值和缩放图像。在训练和推理之间的预处理过程必须相同。

**section 2 - 网络组织**：本节中，我们将介绍3种主要的网络组件，“head”网络、RPN网络和classification网络。

**section 3 - 训练模型的具体实现**：这一节作为该文最长的小节，介绍了训练一个R-CNN网络的具体实现。

**section 3 - 推理的具体实现**：这一节我们将描述推理过程，比如使用训练好了的R-CNN网络去判定兴趣区域（ROI），并在该区域中进行对象的分类。

**附录**：这里我们附加了一些在R-CNN中的常用算法，比如non-maximum-suppression和Resnet 50 architecture。

## 图片的预处理

在下图中的预处理过程在将一张图片送入网络前对该图片进行处理，这些预处理过程在训练和推理过程中必须保持一致。平均向量（3*1，每一个数字对应一个颜色通道）不是指当前图片的像素均值，而是对每一个训练和测试图片一致的 configuration value。

![预处理过程](imgs/img_pre_processing.png)

预定义的参数值是targetSize = 600，maxSize = 1000。

## 网络组织

一个R-CNN网络使用神经网络解决2个主要的问题：

> * 识别一个输入图片的兴趣区域（ROI）内可能包含的前景对象。
> * 计算兴趣区域（ROI）内对象类型的概率分布，比如：计算兴趣区域（ROI）包含某个类别的对象的概率，然后用户可以选择概率最高的对象类别作为分类结果。

R-CNNs主要包含3种网络类型：

1. Head
2. Region Proposal Network (RPN)
3. Classification Network

R-CNNs首先使用前几层网络层作为预训练网络（ResNet 50）在一张输入图片中去识别兴趣特性。由于神经网络具有“转移学习”功能，因此可以使用针对一个不同问题在一个数据集上训练的网络（Yosinski等人，2014）*。前几层网络层的预训练网络学习检测一般的特征，比如边界和颜色斑点（color blobs），这些特征在不同问题中是很好的区别特征。后面几层网络层的学习是更高级别的，针对更多特别问题的特征进行学习。这些网络层可以被移除，或者在反向传播（back-propagation）过程中微调（fine-tuned）这些网络层的权重。从预训练网络初始化前几层网络层构成“head”网络。“head”网络生成卷积特征图然后传递给RPN网络，RPN网络通过多个卷积层和多个全连接层生成兴趣区域（ROIS），该兴趣区域（ROIS）可能包含一个前景对象（(problem 1 mentioned above）。



