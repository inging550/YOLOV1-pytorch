import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
from new_resnet import resnet50


img_root = "***.jpg"   # 需要预测的图片路径
model = resnet50()
model.load_state_dict(torch.load("***.pth"))   # 导入参数
model.eval()
confident = 0.2
iou_con = 0.4

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')  # 将自己的名称输入
CLASS_NUM = len(VOC_CLASSES)   # 20


# target 7*7*30  值域为0-1
class Pred():
    def __init__(self, model, img_root):
        self.model = model
        self.img_root = img_root

    def result(self):
        img = cv2.imread(self.img_root)
        h, w, _ = img.shape
        print(h, w)
        image = cv2.resize(img, (448, 448))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean = (123, 117, 104)  # RGB
        img = img - np.array(mean, dtype=np.float32)
        transform = ToTensor()
        img = transform(img)
        img = img.unsqueeze(0)  # 输入要求是4维的
        Result = self.model(img)   # 1*7*7*30
        bbox = self.Decode(Result)
        bboxes = self.NMS(bbox)    # n*6   bbox坐标是基于7*7网格需要将其转换成448
        for i in range(0, len(bboxes)):    # bbox坐标将其转换为原图像的分辨率
            bboxes[i][0] = bboxes[i][0] * 64
            bboxes[i][1] = bboxes[i][1] * 64
            bboxes[i][2] = bboxes[i][2] * 64
            bboxes[i][3] = bboxes[i][3] * 64

        x1 = bboxes[0][0].item()    # 后面加item()是因为画框时输入的数据不可一味tensor类型
        x2 = bboxes[0][1].item()
        y1 = bboxes[0][2].item()
        y2 = bboxes[0][3].item()
        class_name = bboxes[0][5].item()
        print(x1, x2, y1, y2, VOC_CLASSES[int(class_name)])

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (144, 144, 255))   # 画框
        cv2.imshow('img', image)
        cv2.waitKey(0)

    def Decode(self, result):  # x -> 1**7*30
        result = result.squeeze()   # 7*7*30
        grid_ceil1 = result[:, :, 4].unsqueeze(2)  # 7*7*1
        grid_ceil2 = result[:, :, 9].unsqueeze(2)
        grid_ceil_con = torch.cat((grid_ceil1, grid_ceil2), 2)  # 7*7*2
        grid_ceil_con, grid_ceil_index = grid_ceil_con.max(2)    # 按照第二个维度求最大值  7*7   一个grid ceil两个bbox，两个confidence
        class_p, class_index = result[:, :, 10:].max(2)   # size -> 7*7   找出单个grid ceil预测的物体类别最大者
        class_confidence = class_p * grid_ceil_con   # 7*7   真实的类别概率
        bbox_info = torch.zeros(7, 7, 6)
        for i in range(0, 7):
            for j in range(0, 7):
                bbox_index = grid_ceil_index[i, j]
                bbox_info[i, j, :5] = result[i, j, (bbox_index * 5):(bbox_index+1) * 5]   # 删选bbox 0-5 或者5-10
        bbox_info[:, :, 4] = class_confidence
        bbox_info[:, :, 5] = class_index
        print(bbox_info[1, 5, :])
        return bbox_info  # 7*7*6    6 = bbox4个信息+类别概率+类别代号

    def NMS(self, bbox, iou_con=iou_con):
        for i in range(0, 7):
            for j in range(0, 7):
                # xc = bbox[i, j, 0]        # 目前bbox的四个坐标是以grid ceil的左上角为坐标原点 而且单位不一致
                # yc = bbox[i, j, 1]         # (xc,yc) 单位= 7*7   (w,h) 单位= 1*1
                # w = bbox[i, j, 2] * 7
                # h = bbox[i, j, 3] * 7
                # Xc = i + xc
                # Yc = j + yc
                # xmin = Xc - w/2     # 计算bbox四个顶点的坐标（以整张图片的左上角为坐标原点）单位7*7
                # xmax = Xc + w/2
                # ymin = Yc - h/2
                # ymax = Yc + h/2     # 更新bbox参数  xmin and ymin的值有可能小于0
                xmin = j + bbox[i, j, 0] - bbox[i, j, 2] * 7 / 2     # xmin
                xmax = j + bbox[i, j, 0] + bbox[i, j, 2] * 7 / 2     # xmax
                ymin = i + bbox[i, j, 1] - bbox[i, j, 3] * 7 / 2     # ymin
                ymax = i + bbox[i, j, 1] + bbox[i, j, 3] * 7 / 2     # ymax

                bbox[i, j, 0] = xmin
                bbox[i, j, 1] = xmax
                bbox[i, j, 2] = ymin
                bbox[i, j, 3] = ymax

        bbox = bbox.view(-1, 6)   # 49*6
        bboxes = []
        ori_class_index = bbox[:, 5]
        class_index, class_order = ori_class_index.sort(dim=0, descending=False)
        class_index = class_index.tolist()   # 从0开始排序到7
        bbox = bbox[class_order, :]  # 更改bbox排列顺序
        a = 0
        for i in range(0, CLASS_NUM):
            num = class_index.count(i)
            if num == 0:
                continue
            x = bbox[a:a+num, :]   # 提取同一类别的所有信息
            score = x[:, 4]
            score_index, score_order = score.sort(dim=0, descending=True)
            y = x[score_order, :]   # 同一种类别按照置信度排序
            if y[0, 4] >= confident:    # 物体类别的最大置信度大于给定值才能继续删选bbox，否则丢弃全部bbox
                for k in range(0, num):
                    y_score = y[:, 4]   # 每一次将置信度置零后都重新进行排序，保证排列顺序依照置信度递减
                    _, y_score_order = y_score.sort(dim=0, descending=True)
                    y = y[y_score_order, :]
                    if y[k, 4] > 0:
                        area0 = (y[k, 1] - y[k, 0]) * (y[k, 3] - y[k, 2])
                        for j in range(k+1, num):
                            area1 = (y[j, 1] - y[j, 0]) * (y[j, 3] - y[j, 2])
                            x1 = max(y[k, 0], y[j, 0])
                            x2 = min(y[k, 1], y[j, 1])
                            y1 = max(y[k, 2], y[j, 2])
                            y2 = min(y[k, 3], y[j, 3])
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0 or h < 0:
                                w = 0
                                h = 0
                            inter = w * h
                            iou = inter / (area0 + area1 - inter)
                            # iou大于一定值则认为两个bbox识别了同一物体删除置信度较小的bbox
                            # 同时物体类别概率小于一定值则认为不包含物体
                            if iou >= iou_con or y[j, 4] < confident:
                                y[j, 4] = 0
                for mask in range(0, num):
                    if y[mask, 4] > 0:
                        bboxes.append(y[mask])
            a = num + a
        return bboxes


if __name__ == "__main__":
    Pred = Pred(model, img_root)
    Pred.result()

