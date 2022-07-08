# YOLOV1-pytorch
使用的pytorch版本为1.7.1

主干网络替换为了改进的ResNet50，对应博客地址https://blog.csdn.net/ing100/article/details/125155065

VOC2007数据集下载地址：
链接：https://pan.baidu.com/s/1aB1rXpKlWlEd4WFpBAmzWA 
提取码：1234

步骤：
1、下载数据集，也可以使用自己制作的VOC格式的数据集（只需要将jpg文件以及xml文件分别放入VOCdevkit\VOC2007\中的JPEGImages以及Annotations文件夹内）

2、运行write_txt.py生成训练以及测试需要用到的文本文件

3、运行train.py开始训练

补充：
1、yolov1需要的输入与输出与resnet50不一致，所以此网络结构与原本的resnet50并不完全相同。

2、resnet50.py以及new_resnet.py两者的网络结构都是一样的，但是两者的实现方法不同并且都能够使用。

3、若要使用自己的数据集进行训练需要修改write_txt.py内的类别种类
