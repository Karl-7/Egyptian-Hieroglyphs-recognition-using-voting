import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.preprocessing import OrdinalEncoder
import gc
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score
random.seed(114514)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data=pd.read_csv("C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/train/_annotations.csv")
valid_data = pd.read_csv("C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/valid/_annotations.csv")
test_data = pd.read_csv("C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/test/_annotations.csv")

# 数据文件夹路径
train_folder = "C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/train/"
valid_folder = "C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/valid/"
test_folder = "C:/Users/kjk54/Desktop/LUNWEN/egyptian_csv/test/"
# 定义随机裁剪的变换
crop_transform = transforms.RandomCrop(128)  # 这里的100是裁剪后的图像大小
laplacian_kernel = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])
opening_kernel = np.ones((3, 3), np.uint8)  # 3x3的全1卷积核,用于开运算（先腐蚀后膨胀）
def load_images_from_folder(folder,annotations,crop_flag):
    #添加功能:变为灰度图并裁取文字边缘
    images = []
    gray_transform = transforms.Grayscale()
    for index, row in annotations.iterrows():
        filename=row['filename']
        filename = row['filename']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                #裁切,只取文字部分
                img = img[ymin:ymax, xmin:xmax]
                # 应用卷积操作

                #填补
                if crop_flag == 'True':
                    img = cv2.resize(img,(140,140))
                    img=Laplace_conv_and_adjust_sequence(img,laplacian_kernel,opening_kernel)
                    # 只对训练集图像随即裁切为100x100
                    img = crop_transform(img)
                else:
                    img = cv2.resize(img, (128, 128))
                    img=Laplace_conv_and_adjust_sequence(img,laplacian_kernel,opening_kernel)
                #img== img / 255.0
                images.append(img)
            else:
                continue
    return images

def Laplace_conv_and_adjust_sequence(img,laplacian_kernel,opening_kernel):
    laplace_img = cv2.filter2D(img, cv2.CV_64F, laplacian_kernel)
    # 调整结果范围，将负值置为0，将值限制在0到255之间
    restrict_img = cv2.convertScaleAbs(laplace_img)
    # 进行腐蚀操作
    eroded_image = cv2.erode(restrict_img,opening_kernel, iterations=1)
    # 进行膨胀操作
    dilated_image = cv2.dilate(eroded_image,opening_kernel, iterations=1)

    # cv2.imshow('Original Image', img)
    # cv2.imshow('After Laplace', restrict_img)
    # cv2.imshow('Eroded Image', eroded_image)
    # cv2.imshow('Dilated Image', dilated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # torch的张量是（channels,h,w）,而cv2读入的（h,w,c），要permute换个位置。而训练时格式（batch_size, c ,h,w），所以要添加0 维
    img = torch.tensor(add_gaussian_noise(dilated_image), dtype=torch.float32).unsqueeze(0)  # .permute(3, 1, 2)
    img = img.unsqueeze(1)
    return img
def add_gaussian_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return noisy_image


train_images = load_images_from_folder(train_folder,train_data,'True')
valid_images = load_images_from_folder(valid_folder,valid_data,'False')
test_images = load_images_from_folder(test_folder,test_data,'False')

train_images = torch.cat(train_images, dim=0)
valid_images = torch.cat(valid_images, dim=0)
test_images = torch.cat(test_images, dim=0)
# 对训练集中的图像进行随机裁剪


train_labels = train_data['class'].tolist()
valid_labels = valid_data['class'].tolist()
test_labels = test_data['class'].tolist()

train_labels = np.array(train_labels).reshape(-1,1)
valid_labels = np.array(valid_labels).reshape(-1,1)
test_labels = np.array(test_labels).reshape(-1,1)
#print('after array',test_labels)

encoder = OrdinalEncoder()
encoder.fit(train_labels)
train_labels = encoder.transform(train_labels)
valid_labels = encoder.transform(valid_labels)
test_labels = encoder.transform(test_labels)
#print('after encoder',test_labels)
train_labels = train_labels.flatten()
valid_labels = valid_labels.flatten()
test_labels = test_labels.flatten()
#print('after flatten',test_labels)
#GPU//
train_images=train_images.to(device)
valid_images=valid_images.to(device)
test_images=test_images.to(device)
train_labels = torch.tensor(train_labels, dtype=torch.int64).to(device)
valid_labels = torch.tensor(valid_labels, dtype=torch.int64).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.int64).to(device)
# # #CPU//
# train_images=train_images
# valid_images=valid_images
# test_images=test_images
# train_labels = torch.tensor(train_labels, dtype=torch.int64)
# valid_labels = torch.tensor(valid_labels, dtype=torch.int64)
# test_labels = torch.tensor(test_labels, dtype=torch.int64)

# print('after tensor',train_labels[0:20])
 #                                                                                                     添加这三行灰度值

train_set = TensorDataset(train_images, train_labels)
valid_set = TensorDataset(valid_images, valid_labels)
test_set = TensorDataset(test_images, test_labels)

# num_samples=10
# fig, axes = plt.subplots(num_samples, 1, figsize=(5, 15))
# for i in range(num_samples):
#     sample_index = np.random.randint(len(train_images))
#     sample_image = train_images[sample_index].permute(1, 2, 0).numpy()
#     sample_label = train_labels[sample_index]
#     axes[i].imshow(sample_image)
#     axes[i].set_title(f"Label: {sample_label}")
#     axes[i].axis('off')
#
# plt.tight_layout()
# plt.show()


class_num=len(set(train_labels))
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#                                                                                                       多删几个
del(train_data)
del(valid_data)
del(test_data)
del(train_images)
del(valid_images)
del(train_labels)
del(valid_labels)

gc.collect()
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 打印一些样本图像和标签
num_samples = 5


#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# # I am clearing the RAM from time to time as the memory need is exceeding the available memory :/
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
for epoch in range(70):
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
    #早停
    if valid_accuracy>best_valid_accuracy:
        best_valid_accuracy=valid_accuracy
        best_model_state=model.state_dict()
        patience_counter = 0
    else:
        patience_counter +=1
        if patience_counter >= patience:
            print("Early stopping: No improvement in validation accuracy.")
            break
# # 在测试数据上进行评估

with torch.no_grad():
    test_outputs = model(test_images)
    _, predicted = torch.max(test_outputs, 1)
    print(predicted)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f"TEST准确率：{accuracy}")

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