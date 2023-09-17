import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import cv2
import random
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
import gc
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
opening_kernel = np.ones((2, 2), np.uint8)  # 3x3的全1卷积核,用于开运算（先腐蚀后膨胀）
def load_images_from_folder(folder,annotations,crop_flag):
    #添加功能:变为灰度图并裁取文字边缘
    images = []
    for index, row in annotations.iterrows():
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
    # 拉普拉斯卷积核提取，强化边缘以及图像特征
    gaus_img = add_gaussian_noise(img)
    laplace_img = cv2.filter2D(gaus_img, cv2.CV_64F, laplacian_kernel)
    # 调整结果范围，将负值置为0，将值限制在0到255之间
    restrict_img = cv2.convertScaleAbs(laplace_img)
    # 进行腐蚀操作
    eroded_image = cv2.erode(restrict_img,opening_kernel, iterations=1)
    # 进行膨胀操作
    dilated_image = cv2.dilate(eroded_image,opening_kernel, iterations=1)

    # cv2.imshow('Original Image', img)
    # cv2.imshow('After gauss', gaus_img)
    # cv2.imshow('After Laplace', restrict_img)
    # cv2.imshow('Eroded Image', eroded_image)
    # cv2.imshow('Dilated Image', dilated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # torch的张量是（channels,h,w）,而cv2读入的（h,w,c），要permute换个位置。而训练时格式（batch_size, c ,h,w），所以要添加0 维
    img = torch.tensor(dilated_image, dtype=torch.float32).unsqueeze(0)  # .permute(3, 1, 2)
    img = img.unsqueeze(1)
    return img
def add_gaussian_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = (image + noise)
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

train_set = TensorDataset(train_images, train_labels)
valid_set = TensorDataset(valid_images, valid_labels)
test_set = TensorDataset(test_images, test_labels)

class_num=len(set(train_labels))

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