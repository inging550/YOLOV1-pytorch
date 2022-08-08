# YOLOV1-pytorch
使用的pytorch版本为1.7.1 经测试1.11.0也可以正常运行

主干网络替换为了改进的ResNet50，对应博客地址https://blog.csdn.net/ing100/article/details/125155065

------------------------------------

更新：

2022.8.9 ---- predict.py  部分bug修复


------------------------------------

VOC2007数据集下载地址：
链接：https://pan.baidu.com/s/1aB1rXpKlWlEd4WFpBAmzWA 
提取码：1234

使用VOC2007数据集训练步骤：
1、下载数据集放入文件夹中

2、运行write_txt.py生成训练以及测试需要用到的文本文件

3、运行train.py开始训练

使用自己的训练集训练：

1、将xml文件放入\VOCdevkit\VOC2007\Annotations

2、将jpg文件放入\VOCdevkit\VOC2007\JPEGImages

3、更改write_txt.py中的VOC_CLASSES

4、更改yoloData.py，yoloLoss.py与new_resnet.py中的CLASS_NUM

5、运行write_txt.py生成训练以及测试需要用到的文本文件

6、运行train.py开始训练

补充：
1、yolov1需要的输入与输出与resnet50不一致，所以此网络结构与原本的resnet50并不完全相同。

2、使用VOC2007进行训练需要较大的eppoch大概200左右，若要使用其他数据集进行训练这里给出了个人制作的数据集下载链接在下方

链接：https://pan.baidu.com/s/1tPUhYslsbAMCZxrCJ-GU2Q 

提取码：1234

