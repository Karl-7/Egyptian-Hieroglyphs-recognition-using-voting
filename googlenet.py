import warnings
warnings.filterwarnings('ignore')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import preprocessing as pre
import matplotlib.pyplot as plt
import torch.optim as optim
import time  
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
val_loader=DataLoader(valid_set,batch_size=64,shuffle=True)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(True)
        )
        
        # 1x1 -> 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(True),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(True)
        )
        
        # 1x1 -> 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(True),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(True)
        )
        
        # 3x3 max pooling -> 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, class_num)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建GoogLeNet模型
model = GoogLeNet()

# 将模型移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印模型结构
print(model)

# 将数据移到GPU上

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义学习率
learning_rate = 0.025

# 创建优化器，同时设置学习率
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 创建优化器,学习率自动变化
optimizer = optim.Adam(model.parameters())
# 计算模型的总参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
# 计算并打印模型的总参数数量
total_params = count_parameters(model)
print(f"模型的总参数数量：{total_params}")
# 存储训练历史数据
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}

# 训练循环前记录开始时间
start_time = time.time()
# 训练循环
for epoch in range(70):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        """"# 计算L2正则化项
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param, p=2)  # 求模型参数的L2范数
        
        # 添加L2正则化项到损失函数
        loss += lambda_l2 * l2_regularization"""
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_accuracy = correct_predictions / total_predictions

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    correct_predictions_val = 0
    total_predictions_val = 0
    val_predicted_labels = []  # 存储模型预测的标签
    val_true_labels = []  # 存储真实的标签
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_val = model(inputs_val)
            loss_val = criterion(outputs_val, labels_val)
            val_loss += loss_val.item()

            _, predicted_val = torch.max(outputs_val, 1)
            val_predicted_labels.extend(predicted_val.cpu().numpy())
            val_true_labels.extend(labels_val.cpu().numpy())
            total_predictions_val += labels_val.size(0)
            correct_predictions_val += (predicted_val == labels_val).sum().item()

    val_accuracy = correct_predictions_val / total_predictions_val
    val_loss /= len(val_loader)

    history['loss'].append(running_loss / len(train_loader))
    history['accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(f"第 {epoch+1} 轮，训练损失：{running_loss / len(train_loader)}, 训练准确率：{train_accuracy}")
    print(f"第 {epoch+1} 轮，验证损失：{val_loss}, 验证准确率：{val_accuracy}")
# 训练循环结束后记录结束时间
end_time = time.time()
# 计算整个训练过程的时间
training_time = end_time - start_time
#保存为pkl文本
with open('D:/Jupyter/Egyptian_script/ResNet_training_history_Adam.pkl', 'wb') as f:
    pickle.dump(history, f)
# 在测试数据上进行评估
model.eval()
with torch.no_grad():
    test_outputs = model(test_images)
   # _, predicted = torch.max(test_outputs, 1)
    probabilities = F.softmax(test_outputs, dim=1)
    confidence_scores, predicted = torch.max(probabilities, 1)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    #计算平均置信率
    average_confidence = torch.mean(confidence_scores).item()
    #print(f"测试准确率：{accuracy}")
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_labels.cpu().numpy(), predicted.cpu().numpy())
    print("Confusion Matrix:")
    print(conf_matrix)
    # 设置类别标签
    class_names = [f"Class {i}" for i in range(1,class_num+1)]  # 根据您的类别数量进行设置

   # 计算准确率、置信率和召回率
    test_accuracy = accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
    test_precision = precision_score(test_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    test_recall = recall_score(test_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    # 计算F1分数
    test_f1 = f1_score(test_labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
    print(f"training_time {training_time:.2f} s")
    print(f"test_accuracy：{test_accuracy}")
    print(f"average_confidence：{average_confidence}")
    print(f"test_precision：{test_precision}")
    print(f"test_recall：{test_recall}")
    print(f"test_f1: {test_f1}")

    
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
axs[0, 0].plot(history['loss'], label='Training Loss')
axs[0, 0].plot(history['val_loss'], label='Validation Loss')
axs[0, 0].set_title('GooLeNet Learning Curve')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(history['accuracy'], label='Training Accuracy')
axs[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
axs[0, 1].set_title('GooLeNet Accuracy Curve')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

plt.tight_layout()
plt.show()
# 绘制混淆矩阵图像
plt.figure(figsize=(120, 100))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for GooLeNet")
plt.show()