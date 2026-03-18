import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['CombinedLoss','BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.7 * bce + dice

class EdgeDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 作为边缘检测的损失函数

    def forward(self, input, target):
        # 边缘检测任务的损失
        return self.bce_loss(input, target)


class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, edge_weight=1.0):
        super().__init__()
        self.seg_loss = BCEDiceLoss()  # 分割任务的损失
        self.edge_loss = EdgeDetectionLoss()  # 边缘检测任务的损失
        self.seg_weight = seg_weight  # 分割损失的权重
        self.edge_weight = edge_weight  # 边缘检测损失的权重

    def forward(self, seg_output, edge_output, seg_target, edge_target):
        # 计算分割损失
        seg_loss = self.seg_loss(seg_output, seg_target)
        # 计算边缘检测损失
        edge_loss = self.edge_loss(edge_output, edge_target)
        # 返回加权后的总损失
        return self.seg_weight * seg_loss + self.edge_weight * edge_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss