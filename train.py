import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter
import shutil
import os

# 指定使用的 GPU 设备（例如使用 GPU 1）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCELoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='CombinedLoss', choices=LOSS_NAMES, help='loss: ' + ' | '.join(LOSS_NAMES) +' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='fracdata', help='dataset name')
    #parser.add_argument('--data_dir', default='inputs', help='dataset dir')
    parser.add_argument('--data_dir', default='../../data/slice/dataset/', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'edge_loss': AverageMeter(),  # 新增边缘检测损失
                }


    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, edge_target in train_loader:

        input = input.cuda()
        target = target.cuda()
        edge_target = edge_target.cuda()


        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)

        else:
            # output = model(input)
            # loss = criterion(output, target)
            output, edge_output = model(input)  # 假设模型返回分割结果和边缘检测结果
            loss = criterion(output, edge_output, target, edge_target)  # 使用 CombinedLoss
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        #check_gradients(model)
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['edge_loss'].update(criterion.edge_loss(edge_output, edge_target).item(), input.size(0))  # 更新边缘检测损失


        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('edge_loss', avg_meters['edge_loss'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'edge_loss': AverageMeter()}  # 新增边缘检测损失


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, edge_target in val_loader:
            input = input.cuda()
            target = target.cuda()
            edge_target = edge_target.cuda()


            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                # output = model(input)
                # loss = criterion(output, target)
                output, edge_output = model(input)

                # 计算多任务损失
                loss = criterion(output, edge_output, target, edge_target)  # 使用 CombinedLoss
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['edge_loss'].update(criterion.edge_loss(edge_output, edge_target).item(), input.size(0))  # 更新边缘检测损失


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('edge_loss', avg_meters['edge_loss'].avg)  # 显示边缘检测损失
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('edge_loss', avg_meters['edge_loss'].avg)  # 返回边缘检测损失
                        ])


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
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    # if config['loss'] == 'BCEWithLogitsLoss':
    #     criterion = nn.BCEWithLogitsLoss().cuda()
    # else:
    #     criterion = losses.__dict__[config['loss']]().cuda()

    # 定义损失函数
    if config['loss'] == 'CombinedLoss':
        criterion = losses.CombinedLoss(
            seg_weight=1.0,  # 分割损失的权重
            edge_weight=1.0  # 边缘检测损失的权重
        ).cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'],
                                           embed_dims=config['input_list'], no_kan=config['no_kan'])

    model = model.cuda()

    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower():  # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})

            # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    # elif config['optimizer'] == 'SGD':
    #     optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'],
    #                           weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'
    #
    if dataset_name == 'fracdata':
        mask_ext = '_label.png'


    # Data loading code
    # 训练集和验证集的图像和标签路径
    train_img_ids = sorted(glob(os.path.join(config['data_dir'], 'train', 'images', '*' + img_ext)))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = sorted(glob(os.path.join(config['data_dir'], 'val', 'images', '*' + img_ext)))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]

    # 数据增强和预处理操作
    train_transform = Compose([
            RandomRotate90(),
            geometric.transforms.Flip(),
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
    ],additional_targets={'edge_mask': 'mask'})  # 支持 edge_mask

    val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
    ],additional_targets={'edge_mask': 'mask'})

    # 训练集和验证集的Dataset
    train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(config['data_dir'], 'train', 'images'),
            mask_dir=os.path.join(config['data_dir'], 'train', 'labels'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=train_transform)

    val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(config['data_dir'], 'val', 'images'),
            mask_dir=os.path.join(config['data_dir'], 'val', 'labels'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=val_transform)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('edge_loss', []),  # 新增边缘检测损失
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_edge_loss', []),  # 新增边缘检测损失
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['edge_loss'].append(train_log['edge_loss'])  # 记录训练集的边缘检测损失
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_edge_loss'].append(val_log['edge_loss'])  # 记录验证集的边缘检测损失

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('train/edge_loss', train_log['edge_loss'], global_step=epoch)  # 新增
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/edge_loss', val_log['edge_loss'], global_step=epoch)  # 记录验证集的边缘检测损失

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        epoch_weights_path = f'{output_dir}/{exp_name}/epoch_{epoch}.pth'
        torch.save(model.state_dict(), epoch_weights_path)
        print(f"=> saved epoch {epoch} weights to {epoch_weights_path}")

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()