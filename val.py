#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time
from PIL import Image
import losses


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')

    args = parser.parse_args()

    return args


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    if config['loss'] == 'CombinedLoss':
        criterion = losses.CombinedLoss(
            seg_weight=1.0,  # 分割损失的权重
            edge_weight=1.0  # 边缘检测损失的权重
        ).cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                           embed_dims=config['input_list'])

    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'fracdata':
        mask_ext = '_label.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # 测试集的图像和标签路径
    test_img_ids = sorted(glob(os.path.join(config['data_dir'], 'test', 'images', '*' + img_ext)))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]
    #print(f"Number of test images: {len(test_img_ids)}")
    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')

    try:
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)

    model.eval()

    # 数据增强
    test_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ],additional_targets={'edge_mask': 'mask'})

    # 初始化测试数据集
    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(config['data_dir'], 'test', 'images'),
        mask_dir=os.path.join(config['data_dir'], 'test', 'labels'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=test_transform
    )

    # 初始化测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    edge_loss_avg_meter = AverageMeter()

    with torch.no_grad():
        #for input, target, meta in tqdm(test_loader, total=len(test_loader)):
        for input, target, edge_target, meta in tqdm(test_loader, total=len(test_loader)):

            input = input.cuda()
            target = target.cuda()
            edge_target = edge_target.cuda()
            model = model.cuda()
            # compute output
            output, edge_output = model(input)
            # 计算边缘检测任务的指标
            edge_loss = criterion.edge_loss(edge_output, edge_target)
            edge_loss_avg_meter.update(edge_loss.item(), input.size(0))

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            #for pred, img_id in zip(output, test_img_ids):
            for pred, img_id in zip(output, meta):
                pred_np = pred[0].astype(np.uint8)
                pred_np = (1 - pred_np) * 255  # 由于标签像素为0，预测结果为0的像素变为255（即白色）

                # 加载对应的原始图像
                original_img_path = os.path.join(config['data_dir'], 'test', 'images', f"{img_id}{img_ext}")
                original_img = Image.open(original_img_path).convert("RGB")

                # 将预测结果转换为图像
                pred_img = Image.fromarray(pred_np, 'L')

                # 调整预测图像的大小以匹配原始图像
                pred_img = pred_img.resize(original_img.size, Image.NEAREST)

                # 创建一个红色的RGBA掩码
                red_mask = Image.new("RGBA", pred_img.size, (255, 0, 0, 0))
                red_mask_data = red_mask.load()

                # 设置透明度为0.7
                alpha_value = int(0.7 * 255)

                # 遍历掩码，修改为红色并设置alpha值
                for y in range(pred_img.size[1]):
                    for x in range(pred_img.size[0]):
                        if pred_img.getpixel((x, y)) == 255:  # 仅对标签区域处理
                            red_mask_data[x, y] = (255, 0, 0, alpha_value)

                # 将红色掩码叠加在原始图像上
                combined_img = Image.alpha_composite(original_img.convert("RGBA"), red_mask)

                # 保存叠加后的图像
                combined_img = combined_img.convert("RGB")  # 转回RGB格式保存为jpg
                combined_img.save(
                    os.path.join(args.output_dir, config['name'], 'out_val/{}_overlay.jpg'.format(img_id)))

                # 同时保存预测的标签图像
                pred_img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)


if __name__ == '__main__':
    main()
