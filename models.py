import torch
import torch.nn as nn          # PyTorch的神经网络模块

# 自定义神经网络模型
class SimpleNN(nn.Module):     # 自定义模型继承自nn.Module（PyTorch所有模型的基类）
    def __init__(self):        # 初始化函数，定义网络层
        super().__init__()     # 继承父类的初始化
        # 定义全连接层（也叫线性层，nn.Linear），是最基础的神经网络层，作用是对输入数据做线性变换（类似 y = wx + b，w 是权重，b 是偏置）
        self.fc1 = nn.Linear(28*28, 256) # 全连接层1，输入大小为28*28，输出大小为256（ MNIST数据集的图片是28x28像素的灰度图，展平后就是一个长度为28*28的一维向量）
        self.fc2 = nn.Linear(256, 128)   # 全连接层2，输入大小为256，输出大小为128(输入必须和上一层输出一致)
        self.fc3 = nn.Linear(128, 64)    # 全连接层3，输入大小为128，输出大小为64      （中间层的256、128等是隐藏层的神经元熟练，是认为设定的，通过多层变换，学习到更复杂的特征）
        self.fc4 = nn.Linear(64, 10)     # 全连接层4，输入大小为64，输出大小为10（对应10个类别）

    def forward(self, x):                  # 前向传播函数，定义输入数据x如何流过网络
        x = torch.flatten(x,start_dim=1)   # 展平输入，维度start_dim=1表示从第1维开始展平，保持第0维（批次大小）不变,输入的图片格式是 (批量大小, 通道数, 高度, 宽度) 
        x = torch.relu(self.fc1(x))        # 通过全连接层1，再应用ReLU激活函数（把负数变成 0，正数不变，给网络增加 “非线性”，让它能学习更复杂的规律）
        x = torch.relu(self.fc2(x))        # 通过全连接层2，再应用ReLU激活函数
        x = torch.relu(self.fc3(x))        # 通过全连接层3，再应用ReLU激活函数
        x = self.fc4(x)                    # 通过全连接层4（输出10个数字，不需要激活函数，后续计算损失时内部包含了softmax操作，进行概率计算）
        return x