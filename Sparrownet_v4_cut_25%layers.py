import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import preprocessing as pre
import time  
#import seaborn as sns
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess
import re
from collections import Counter
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
batch_size=64
train_loader=DataLoader(train_set,batch_size,shuffle=True)
val_loader=DataLoader(valid_set,batch_size,shuffle=True)

class BasicBlock(nn.Module):
    #添加2x1卷积用到的缩减后的维度out_3x3_reduce
    def __init__(self, in_channels, out_channels,out_3x3_reduce, stride=1):
        super(BasicBlock, self).__init__()
        self.branch1=nn.Sequential(
            #添加1x1卷积层降维
            ###################
            nn.Conv2d(in_channels, out_3x3_reduce,kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(True),
            ###################
            nn.Conv2d(out_3x3_reduce, out_3x3_reduce, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_3x3_reduce)
        )
        self.branch2=nn.Sequential(
            #添加1x1卷积层降维
            ###################
            nn.Conv2d(out_3x3_reduce, out_3x3_reducekernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(True),
            ###################
            nn.Conv2d(out_3x3_reduce, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        # 如果输入和输出的通道数不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_3x3_reduce),
                #添加1x1卷积层降维
            ###################
                nn.Conv2d(out_3x3_reduce, out_channels,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
         # 添加Dropout层
        #self.dropout = nn.Dropout(p=0.5)  # 50%的概率随机置零
        
    def forward(self, x):
        out = self.branch1(x)
        #print("branch1",out.shape)
        out = self.branch2(out)
        #print("branch2",out.shape)
        out += self.shortcut(x)
        out = F.relu(out)
        #out = self.dropout(out)  # 在模型中应用Dropout层
        return out


class Sparrownet(nn.Module):
    def __init__(self, num_classes=class_num):
        super(Sparrownet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 32, 2)
        self.layer2 = self._make_layer(64, 128, 32, 2, stride=2)
        self.layer3 = self._make_layer(128, 256,96, 2, stride=2)
        #self.layer4 = self._make_layer(256, 512,128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels,out_3x3_reduce, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, out_3x3_reduce, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, out_3x3_reduce))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# model = Sparrownet()

MAGI=3
sparrows=[Sparrownet().to(device) for i in range(MAGI)]

#optimizer=torch.optim.Adam([{"params":model.parameters()} for model in sparrows],lr=0.01)
#由Adam定义LR
optimizer=torch.optim.Adam([{"params":sparrow.parameters()} for sparrow in sparrows])

# 打印模型结构
# print(model)

# 定义损失函数
criterion = nn.CrossEntropyLoss()


# 创建优化器，同时设置学习率
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 计算模型的总参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
# 计算并打印模型的总参数数量
total_params = count_parameters(sparrows[0])
print(f"单个模型的参数量：{total_params}")
# 存储训练历史数据
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}

# 存储GPU占用率历史数据的列表
gpu_utilization_history = []
# 训练循环前记录开始时间
start_time = time.time()
# 训练循环
for epoch in range(70):
     # 获取GPU占用率#####################################################################################
    nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
    nvidia_smi_output = nvidia_smi_output.decode('utf-8')
    gpu_utilization_lines = nvidia_smi_output.strip().split('\n')
    gpu_utilization = [int(re.sub(r'\D', '', line)) for line in gpu_utilization_lines]
    gpu_utilization_history.extend(gpu_utilization)
    average_gpu_utilization = sum(gpu_utilization_history) / len(gpu_utilization_history)
    ########################################################################################################
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        for i,model in enumerate(sparrows):
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            """# 计算L2正则化项
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
            correct_predictions += (predicted == labels).sum().item()

        # temp_arr=np.array(pre)
        # result=[Counter(temp_arr[:,MAGI]).most_common(1)[0][0] for i in range(batch_size)]
        total_predictions += labels.size(0)
    train_accuracy = correct_predictions / total_predictions/MAGI
    #CORRECT_PREDICTIONS是所有3个投票器的总正确分类数量，所以要除以3（MAGI=3)

    # 在验证集上评估模型
    # model.eval()
    val_loss = 0.0
    correct_predictions_val = 0
    total_predictions_val = 0
    with torch.no_grad():
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            for i,model in enumerate(sparrows):
                model.eval()
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val, 1)
                correct_predictions_val += (predicted_val == labels_val).sum().item()
            total_predictions_val += labels_val.size(0)
    val_accuracy = correct_predictions_val / total_predictions_val/MAGI
     #将GPU占用率历史数据添加到history字典中
    history['gpu_utilization'] = gpu_utilization_history
    history['loss'].append(running_loss / len(train_loader))
    history['accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(f"第 {epoch+1} 轮，训练损失：{running_loss / len(train_loader) / MAGI}, 训练准确率：{train_accuracy}")
    print(f"第 {epoch+1} 轮，验证损失：{val_loss / len(val_loader) / MAGI}, 验证准确率：{val_accuracy}")
# 训练循环结束后记录结束时间
end_time = time.time()
# 计算整个训练过程的时间
training_time = end_time - start_time
#保存为pkl文本
with open('C:/Users/kjk54/Desktop/egypt/egyptian_csv/SparrowNet_V2_training_history.pkl', 'wb') as f:
    pickle.dump(history, f)





# 在测试数据上进行评估
# 在测试集上运行3次，记录投票结果
test_predicted_labels = []
with torch.no_grad():
    for i ,model in enumerate(sparrows):
        model.eval()
        test_outputs = model(test_images)
        probabilities = F.softmax(test_outputs, dim=1)
        confidence_scores, predicted = torch.max(probabilities, 1)
        test_predicted_labels.append(predicted.cpu().numpy())
     #计算平均置信率
    average_confidence = torch.mean(confidence_scores).item()
    test_predicted_labels = np.array(test_predicted_labels)
    # 计算投票结果
    test_predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=test_predicted_labels)
    # 计算准确率、置信率和召回率
    # 计算平均GPU占用率
    average_gpu_utilization = sum(gpu_utilization) / len(gpu_utilization)
    test_accuracy = accuracy_score(test_labels.cpu().numpy(), test_predicted_labels)
    test_precision = precision_score(test_labels.cpu().numpy(), test_predicted_labels, average='weighted')
    test_recall = recall_score(test_labels.cpu().numpy(), test_predicted_labels, average='weighted')
    # 计算F1分数
    test_f1 = f1_score(test_labels.cpu().numpy(), test_predicted_labels, average='weighted')
    print(f"training_time {training_time:.2f} s")
    print(f"test_accuracy after voting：{test_accuracy}")
    print(f"average_confidence after voting：{average_confidence}")
    print(f"test_precision after voting：{test_precision}")
    print(f"test_recall after voting：{test_recall}")
    print(f"test_f1 after voting：{test_f1}")
    print(f"average_gpu_utilization：{average_gpu_utilization:.2f}%")



    # 绘制学习曲线

fig, axs = plt.subplots(2, 2, figsize=(20, 15))
axs[0, 0].plot(history['loss'], label='Training Loss')
axs[0, 0].plot(history['val_loss'], label='Validation Loss')
axs[0, 0].set_title('ResNet Learning Curve')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(history['accuracy'], label='Training Accuracy')
axs[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
axs[0, 1].set_title('ResNet Accuracy Curve')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

plt.tight_layout()
plt.show()
# 绘制混淆矩阵图像
plt.figure(figsize=(120, 100))
#sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for ResNet")
plt.show()