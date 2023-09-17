import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import preprocessing as pre
import random
import pickle
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
random.seed(114514)
device=pre.device

train_set = pre.train_set
valid_set = pre.valid_set
test_set = pre.test_set

test_labels=pre.test_labels
test_images=pre.test_images
class_num=pre.class_num
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# 添加 权重采样、模型正则化
# Weighted Sampling
# class_weights = [1.0] * class_num  # Initialize with equal weights
# #class_weights = calculate_class_weights(train_set[train_labels])# Calculate class weights for weighted sampling
# weighted_sampler = WeightedRandomSampler(class_weights, len(train_set[train_labels]), replacement=True)
# train_loader=DataLoader(train_set,batch_size=16,shuffle=True,sampler=weighted_sampler)
#由于train_folder分布均匀，不需要权重采样
train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
val_loader=DataLoader(valid_set,batch_size=64,shuffle=True)

# 在 PyTorch 中定义模型
# 定义ResNet基本块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 如果输入和输出的通道数不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 64, num_blocks[2], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# 创建ResNet18模型
#GPU//
model = ResNet(BasicBlock, [2, 2, 2],class_num).to(device)
#CPU//
#model = ResNet(BasicBlock, [2, 2, 2],class_num)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5)

# 存储训练历史数据
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}
#添加 早停 Early Stopping
best_valid_accuracy = 0.0
best_model_state = None
patience = 30
# 训练循环
for epoch in range(80):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        #print('这啥啊这是',inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    # 计算验证集上的损失和准确率
    valid_loss = 0.0
    correct_predictions_valid = 0
    total_predictions_valid = 0

    model.eval()  # 将模型切换到评估模式
    with torch.no_grad():
        for inputs_valid, labels_valid in val_loader:  # 假设你有一个验证集的 DataLoader
            outputs_valid = model(inputs_valid)
            loss_valid = criterion(outputs_valid, labels_valid)
            valid_loss += loss_valid.item()

            _, predicted_valid = torch.max(outputs_valid, 1)
            total_predictions_valid += labels_valid.size(0)
            correct_predictions_valid += (predicted_valid == labels_valid).sum().item()

    valid_accuracy = correct_predictions_valid / total_predictions_valid
    valid_loss /= len(val_loader)

    # 将当前 epoch 的训练和验证数据记录到 history 字典中
    history['loss'].append(running_loss / len(train_loader))
    history['accuracy'].append(correct_predictions / total_predictions)
    history['val_loss'].append(valid_loss)
    history['val_accuracy'].append(valid_accuracy)

    print(f"第 {epoch + 1} 轮，损失：{running_loss / len(train_loader)}")
    print(f"第 {epoch + 1} 轮，验证损失：{valid_loss}, 验证准确率：{valid_accuracy}")
    # #早停
    # if valid_accuracy>best_valid_accuracy:
    #     best_valid_accuracy=valid_accuracy
    #     best_model_state=model.state_dict()
    #     patience_counter = 0
    # else:
    #     patience_counter +=1
    #     if patience_counter >= patience:
    #         print("Early stopping: No improvement in validation accuracy.")
    #         break
# # 在测试数据上进行评估
    with open('ResNet_lr=0.01','wb') as file:
        pickle.dump(history,file)
with torch.no_grad():
    test_outputs = model(test_images)
    _, predicted = torch.max(test_outputs, 1)
    print(predicted)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f"TEST准确率：{accuracy}")

# 绘制学习曲线

fig, axs = plt.subplots(2, 1, figsize=(20, 15))
axs[0].plot(history['loss'], label='Training Loss')
axs[0].plot(history['val_loss'], label='Validation Loss')
axs[0].set_title('ResNet Learning Curve')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(history['accuracy'], label='Training Accuracy')
axs[1].plot(history['val_accuracy'], label='Validation Accuracy')
axs[1].set_title('ResNet Accuracy Curve')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.show()