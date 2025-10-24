from torchvision import datasets,transforms   # torchvision.datasets包含常用的图像数据集（比如MNIST手写数字数据集）；torchvision.transforms是图像预处理工具，比如缩放、转张量、归一化等
from torch.utils.data import DataLoader       # 批量加载数据的工具
import matplotlib.pyplot as plt               # 可视化绘图工具
from models import SimpleNN                    # 导入自定义的神经网络模型
import torch

transform = transforms.Compose([                    # 图像预处理的组合操作
    transforms.Grayscale(num_output_channels=1),    # 转为灰度图，通道数为1
    transforms.ToTensor(),                          # 将图像转换为张量，像素值变为0-1(原始图像像素值为0-255)
    transforms.Normalize([0.5], [0.5])              # 归一化，标准化到[-1, 1]，公式：(x - 均值) / 标准差，前一个[0.5] 是均值，后一个[0.5] 是标准差
])
# 加载MNIST数据集
train_dataset = datasets.MNIST(          # 加载MNIST训练数据集
    root='./data',                       # 数据集存放路径
    train=True,                          # 加载训练集
    transform=transform,                 # 应用预处理操作
    download=True                        # 如果数据集不存在则下载
)
# 加载MNIST测试数据集
test_dataset = datasets.MNIST(            # 加载MNIST测试数据集
    root='./data',                        # 数据集存放路径
    train=False,                          # 加载测试集
    transform=transform                   # 应用预处理操作
)
# 创建数据加载器
train_loader = DataLoader(               # 创建训练数据加载器
    dataset=train_dataset,               # 传入训练数据集
    batch_size=64,                       # 每个批次64张图像
    shuffle=True                         # 每个epoch打乱数据（epoch指数据集被完整遍历一次的过程）
)
# 创建测试数据加载器
test_loader = DataLoader(                 # 创建测试数据加载器
    dataset=test_dataset,                 # 传入测试数据集
    batch_size=64,                        # 每个批次64张图像
    shuffle=False                         # 不打乱数据
)

import torch.nn as nn            # PyTorch的神经网络模块
import torch.optim as optim      # 优化器模块，包含各种优化算法
from tqdm import tqdm            # 进度条显示工具

#检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#初始化模型
model = SimpleNN().to(device)   # 实例化自定义的神经网络模型，并将其移动到指定设备（GPU或CPU）

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()          # 交叉熵损失函数(错的越离谱值就越大)，常用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器（根据损失函数的结果，调整模型线性变换里的权重w和偏置b，学习率0.001是调整幅度）

#用于保存训练过程中的损失和准确率
train_losses = []
train_accuracies = []
test_accuracies = []

#训练模型
num_epochs = 10                          # 训练轮数
best_accuracy = 0.0                      # 用于保存最佳测试准确率
best_model_path = 'best_model.pth'       # 保存最佳模型的路径

for epoch in range(num_epochs):
    running_loss = 0.0                   # 累计本轮的损失
    correct_train = 0                    # 训练集上正确预测的样本数
    total_train = 0                      # 训练集上总样本数

    model.train()                        # 设置模型为训练模式

    # 用tqdm显示进度条（desc参数设置进度条前缀），遍历训练集中的每一批数据
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备
       
        optimizer.zero_grad()            # 清零梯度
        outputs = model(images)          # 前向传播，模型对图片做预测
        loss = criterion(outputs, labels)# 计算损失
        loss.backward()                  # 反向传播
        optimizer.step()                 # 更新参数

        running_loss += loss.item()      # 累计损失

        _, predicted = torch.max(outputs, 1)     # 获取预测结果（_为最大值，predicted取最大值索引即预测数字），模型输出的二维张量格式是【batch_size,类别维度】，取1代表类别维度（概率）
        total_train += labels.size(0)            # 累计样本数
        correct_train += (predicted == labels).sum().item()  # 累计正确预测数

    # 计算训练集上的准确率
    train_accuracy = correct_train / total_train
    train_losses.append(running_loss / len(train_loader))  # 计算并保存本轮的平均损失
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy*100:.2f}%')

    # 训练完一轮后，在测试集上评估模型
    model.eval()                     # 设置模型为评估模式
    correct_test = 0                 # 测试集上正确预测的样本数
    total_test = 0                   # 测试集上总样本数
    with torch.no_grad():               # 评估时不计算梯度
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        
    test_accuracy = correct_test / total_test       # 计算测试集上的准确率
    test_accuracies.append(test_accuracy)
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)     # 保存模型参数(权重和偏置)
        print(f'Best model saved with accuracy: {best_accuracy*100:.2f}%')

print(f'Best Test Accuracy over all epochs: {best_accuracy*100:.2f}%')

# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')  # 保存图像到文件
plt.show()