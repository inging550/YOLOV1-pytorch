import xml.etree.ElementTree as ET
import os
import random

VOC_CLASSES = (  # 定义所有的类名
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# 定义一些参数
train_set = open('voctrain.txt', 'w')
test_set = open('voctest.txt', 'w')
Annotations = 'VOCdevkit//VOC2007//Annotations//'
xml_files = os.listdir(Annotations)
random.shuffle(xml_files)  # 打乱数据集
train_num = int(len(xml_files) * 0.7)  # 训练集数量
train_lists = xml_files[:train_num]   # 训练列表
test_lists = xml_files[train_num:]    # 测测试列表


def parse_rec(filename):  # 输入xml文件名
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:  # 若为1则跳过本次循环
            continue
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects


def write_txt():
    count = 0
    for train_list in train_lists: # 生成训练集txt
        count += 1
        image_name = train_list.split('.')[0] + '.jpg'  # 图片文件名
        results = parse_rec(Annotations + train_list)
        if len(results) == 0:
            print(train_list)
            continue
        train_set.write(image_name)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            train_set.write(' ' + str(bbox[0]) +
                            ' ' + str(bbox[1]) +
                            ' ' + str(bbox[2]) +
                            ' ' + str(bbox[3]) +
                            ' ' + str(class_name))
        train_set.write('\n')
    train_set.close()

    for test_list in test_lists:   # 生成测试集txt
        count += 1
        image_name = test_list.split('.')[0] + '.jpg'  # 图片文件名
        results = parse_rec(Annotations + test_list)
        if len(results) == 0:
            print(test_list)
            continue
        test_set.write(image_name)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)
            test_set.write(' ' + str(bbox[0]) +
                            ' ' + str(bbox[1]) +
                            ' ' + str(bbox[2]) +
                            ' ' + str(bbox[3]) +
                            ' ' + str(class_name))
        test_set.write('\n')
    test_set.close()


if __name__ == '__main__':
    write_txt()
