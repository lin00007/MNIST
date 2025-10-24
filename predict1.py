import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image      # 用于读取自定义图片（MNIST数据集用datasets自动读，自定义图需手动读）
from models import SimpleNN
import matplotlib.pyplot as plt     # 用于显示图片

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 定义图像预处理（和训练时保持一致）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # 转为单通道灰度图
    transforms.Resize((28, 28)),                   # 调整图像大小为28x28像素(MNIST数据集的标准尺寸)
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 加载训练好的模型
model = SimpleNN().to(device)     # 实例化模型并放入设备
model.load_state_dict(torch.load('best_model.pth', map_location=device))  # 加载模型参数（map_location=device 确保参数加载到指定设备）
model.eval()                      # 设置模型为评估模式,关闭训练特有的层

# 读取并处理单张图片
img_path = r'./myfigure2.jpg'   # 要推理的图片路径
image = Image.open(img_path)   # 使用PIL打开图片
image = transform(image).unsqueeze(0).to(device)  # 图片预处理+添加批次维度+放入设备（模型中的图片格式为B C H W是一个四维的张量，而图片格式是C H W,Batch size默认为1）

# 进行预测
with torch.no_grad():           # 关闭梯度计算，节省内存和计算
    output = model(image)       # 将处理后的图片输入模型进行推理
    _, predicted = torch.max(output, 1)  # 获取预测结果
    print(f'Predicted class: {predicted.item()}')  # 打印预测的类别

# 展示图片和预测类别
plt.imshow(Image.open(img_path), cmap='gray')      # 读取原始图片（用于显示）
plt.title(f'Predicted class: {predicted.item()}')  #  给图片加标题，显示预测结果
plt.axis('off')  # 关闭坐标轴
plt.show()  # 展示图片