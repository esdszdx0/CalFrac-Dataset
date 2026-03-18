# model_summary.py
import torch
from torchinfo import summary
from archs import UKAN  # 导入自定义模型
from fvcore.nn import FlopCountAnalysis


num_classes = 1
# 创建模型
model = UKAN(num_classes = num_classes)

# 设置输入的尺寸
input_size = (1, 3, 224, 224)  # 输入为3通道224x224的图像
input_tensor = torch.randn(*input_size).cuda()  # 创建一个输入张量


# 使用torchinfo显示模型的概况
summary(model, input_size=input_size, verbose=1)

flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")  # 输出总的FLOPs数
