import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.*'))  # 匹配所有文件格式
        print(f"加载到的图片路径: {self.imgs_path}")  # 调试代码

    def augment(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip

    def __getitem__(self, index):
        # 读取图片和标签
        image_path = self.imgs_path[index]
        label_path = image_path.replace('train_image', 'train_label')  # 确保替换逻辑正确
        print(f"加载到的图片路径: {image_path}, 标签路径: {label_path}")  # 调试代码

        image = cv2.imread(image_path)  # RGB 3通道图片
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # 对图片进行预处理
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        # 图像增强
        flipcode = random.choice([-1, 0, 1, 2])
        if flipcode != 2:
            image = self.augment(image, flipcode)
            label = self.augment(label, flipcode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    data_path = r"train_image"
    plate_dataset = SelfDataSet(data_path)
    print(f"数据集大小: {len(plate_dataset)}")  # 调试代码
    train_loader = torch.utils.data.DataLoader(dataset=plate_dataset, batch_size=5, shuffle=True)
    for image, label in train_loader:
        print(f"图片批次大小: {image.shape}, 标签批次大小: {label.shape}")