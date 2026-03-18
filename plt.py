import pandas as pd
import matplotlib.pyplot as plt

# 读取不同权重下的CSV文件
df_weight_01 = pd.read_csv('./outputs/None/log2.csv')  # 权重0.1
df_weight_02 = pd.read_csv('./outputs/None/log5.csv')  # 权重0.5

# 删除包含零值的行
df_weight_01_clean = df_weight_01[(df_weight_01[['epoch', 'loss', 'val_loss', 'iou', 'val_iou', 'val_dice']] > 0).all(axis=1)]
df_weight_02_clean = df_weight_02[(df_weight_02[['epoch', 'loss', 'val_loss', 'iou', 'val_iou', 'val_dice']] > 0).all(axis=1)]
# 检查清洗后的数据
print("Cleaned data for weight 0.3:")
print(df_weight_01_clean.head())
print("Cleaned data for weight 0.5:")
print(df_weight_02_clean.head())

# 获取数据（根据文件的列名进行调整）
epochs_01 = df_weight_01['epoch']
train_loss_01 = df_weight_01['loss']
val_loss_01 = df_weight_01['val_loss']
train_iou_01 = df_weight_01['iou']
val_iou_01 = df_weight_01['val_iou']
# train_dice_01 = df_weight_01['iou']  # 假设用 IoU 代替 Dice
val_dice_01 = df_weight_01['val_dice']

epochs_02 = df_weight_02['epoch']
train_loss_02 = df_weight_02['loss']
val_loss_02 = df_weight_02['val_loss']
train_iou_02 = df_weight_02['iou']
val_iou_02 = df_weight_02['val_iou']
# train_dice_02 = df_weight_02['iou']  # 假设用 IoU 代替 Dice
val_dice_02 = df_weight_02['val_dice']
# print("Checking data validity for weight 0.3:")
# print(df_weight_01[['epoch', 'loss', 'val_loss', 'iou', 'val_iou', 'val_dice']].isnull().sum())
# print("Checking data validity for weight 0.5:")
# print(df_weight_02[['epoch', 'loss', 'val_loss', 'iou', 'val_iou', 'val_dice']].isnull().sum())

# 绘制损失曲线
plt.figure(figsize=(12, 8))

# 权重0.3
plt.plot(epochs_01, train_loss_01, label='Train Loss (Weight=0.3)', color='blue', linestyle='-', marker='o')
plt.plot(epochs_01, val_loss_01, label='Validation Loss (Weight=0.3)', color='red', linestyle='--', marker='x')

# 权重0.5
plt.plot(epochs_02, train_loss_02, label='Train Loss (Weight=0.5)', color='green', linestyle='-', marker='o')
plt.plot(epochs_02, val_loss_02, label='Validation Loss (Weight=0.5)', color='orange', linestyle='--', marker='x')

plt.title('Comparison of Training and Validation Loss for Different Weights')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制IoU曲线
plt.figure(figsize=(12, 8))

# 权重0.3
plt.plot(epochs_01, train_iou_01, label='Train IoU (Weight=0.3)', color='blue', linestyle='-', marker='o')
plt.plot(epochs_01, val_iou_01, label='Validation IoU (Weight=0.3)', color='red', linestyle='--', marker='x')

# 权重0.5
plt.plot(epochs_02, train_iou_02, label='Train IoU (Weight=0.5)', color='green', linestyle='-', marker='o')
plt.plot(epochs_02, val_iou_02, label='Validation IoU (Weight=0.5)', color='orange', linestyle='--', marker='x')

plt.title('Comparison of Training and Validation IoU for Different Weights')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.grid(True)
plt.show()

# 绘制Dice系数曲线
plt.figure(figsize=(12, 8))

# 权重0.1
#plt.plot(epochs_01, train_dice_01, label='Train Dice (Weight=0.3)', color='blue', linestyle='-', marker='o')
plt.plot(epochs_01, val_dice_01, label='Validation Dice (Weight=0.3)', color='red', linestyle='--', marker='x')

# 权重0.5
#plt.plot(epochs_02, train_dice_02, label='Train Dice (Weight=0.5)', color='green', linestyle='-', marker='o')
plt.plot(epochs_02, val_dice_02, label='Validation Dice (Weight=0.5)', color='orange', linestyle='--', marker='x')

plt.title('Comparison of Training and Validation Dice for Different Weights')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.grid(True)
plt.show()

# 额外分析（比较两个权重的最终表现）
plt.figure(figsize=(10, 6))

# 绘制最终的性能对比（例如：选择最后一个epoch的结果）
final_epoch_01 = epochs_01.iloc[-1]  # 获取最后一轮的epoch
final_loss_01 = val_loss_01.iloc[-1]
final_iou_01 = val_iou_01.iloc[-1]
final_dice_01 = val_dice_01.iloc[-1]

final_epoch_02 = epochs_02.iloc[-1]  # 获取最后一轮的epoch
final_loss_02 = val_loss_02.iloc[-1]
final_iou_02 = val_iou_02.iloc[-1]
final_dice_02 = val_dice_02.iloc[-1]

# 绘制最终损失
plt.bar([final_epoch_01, final_epoch_02], [final_loss_01, final_loss_02], label='Loss', color='blue', alpha=0.6)
# 绘制最终IoU
plt.bar([final_epoch_01, final_epoch_02], [final_iou_01, final_iou_02], label='IoU', color='green', alpha=0.6)
# 绘制最终Dice
plt.bar([final_epoch_01, final_epoch_02], [final_dice_01, final_dice_02], label='Dice', color='orange', alpha=0.6)

plt.title('Final Performance Comparison for Different Weights')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()
