import warnings
warnings.filterwarnings('ignore')
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import preprocessing as pre

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
class CustomCNN(nn.Module):
    def __init__(self, num_conv_layers, num_classes):
        super(CustomCNN, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.num_classes = num_classes
        self.layers = self._create_layers()
        # Calculate the size of the linear layer input
        self.fc_input_size = self.calculate_conv_output_size(16, 3, 1, 1)  # Assuming conv3's parameters
        self.fc1 = nn.Linear(128 * self.fc_input_size * self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, class_num)

    def forward(self, x):
        #print("Input shape:", x.shape)
        for layer in self.layers:
            x = layer(x)


        #print("Shape before reshape:", x.shape)
        # Reshape the tensor before passing it through fully connected layers
        x = x.view(-1, 128 * self.fc_input_size * self.fc_input_size)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.5)
        # x = self.fc3(x)
        return x

    # Function to calculate the output size of convolutional layers
    def calculate_conv_output_size(self, image_size, kernel_size, padding, stride):
        return ((image_size - kernel_size + 2 * padding) // stride) + 1

    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = 1  # Input channel for the first conv layer
        out_channels = 32  # Initial number of output channels

        for _ in range(self.num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2  # Double the output channels in each layer

        return layers

# Instantiate your model with 50 conv layers and 3 fully connected layers
num_conv_layers = 3
#GPU//
model = CustomCNN(num_conv_layers, class_num).to(device)
#CPU//
# model = CustomCNN(num_conv_layers, class_num)

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
patience = 5
# 训练循环
for epoch in range(100):
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
# 在测试数据上进行评估

with torch.no_grad():
    test_outputs = model(test_images)
    _, predicted = torch.max(test_outputs, 1)
    #print(predicted)
    correct = (predicted == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f"TEST准确率：{accuracy}")

# 绘制学习曲线

fig, axs = plt.subplots(2, 2, figsize=(20, 15))
axs[0, 0].plot(history['loss'], label='Training Loss')
axs[0, 0].plot(history['val_loss'], label='Validation Loss')
axs[0, 0].set_title('CustomCNN Learning Curve')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()

axs[0, 1].plot(history['accuracy'], label='Training Accuracy')
axs[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
axs[0, 1].set_title('CustomCNN Accuracy Curve')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].legend()

plt.tight_layout()
plt.show()