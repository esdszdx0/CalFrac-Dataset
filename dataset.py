import os

import cv2
import numpy as np
import torch
import torch.utils.data
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generate_edge_mask(mask):
    """
    从分割标签生成边缘标签。
    :param mask: 分割标签，形状为 [H, W]，值为 0 或 1。
    :return: 边缘标签，形状为 [H, W]，值为 0 或 1。
    """
    mask = mask.astype(np.uint8) * 255  # 将 mask 转换为 0-255 范围
    edges = cv2.Canny(mask, threshold1=10, threshold2=100)  # 使用 Canny 边缘检测
    edges = (edges > 0).astype(np.float32)  # 将边缘转换为二值标签
    return edges

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # Load mask directly from the '0' folder
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]

        # 生成边缘标签
        edge_mask = generate_edge_mask(mask)


        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask, edge_mask=edge_mask)
            img = augmented['image']
            mask = augmented['mask']
            edge_mask = augmented['edge_mask']


        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        edge_mask = edge_mask.astype('float32')  # 形状为 [H, W]
        edge_mask = np.expand_dims(edge_mask, axis=0)  # 形状为 [1, H, W]

        if mask.max() < 1:
            mask[mask > 0] = 1.0

        return img, mask, edge_mask

        #return img, mask, edge_mask, img_id


