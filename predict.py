import cv2           # 用于图像显示和处理
import torch         # PyTorch深度学习框架核心库
from torchvision import datasets,transforms     # 数据集和图像预处理工具
from torch.utils.data import DataLoader         # 批量加载数据的工具
from models import SimpleNN                     # 导入自定义的神经网络模型

transform = transforms.Compose([                # 图像预处理的组合操作
    transforms.Grayscale(num_output_channels=1),    
    transforms.ToTensor(),                          
    transforms.Normalize([0.5], [0.5])              
])

# 加载MNIST测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
# 创建测试集的数据加载器
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=True
)

model = SimpleNN()         # 实例化模型结构
model.load_state_dict(torch.load('best_model.pth'))     # 加载训练好的模型参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)           # 将模型移动到指定设备（GPU或CPU）

model.eval()               # 设置模型为评估模式
with torch.no_grad():      # 关闭梯度计算
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)                   # 模型推理：输入图像，输出预测结果
        _, predicted = torch.max(outputs.data, 1) # 从输出概率中取最大值对应的索引，_是最大值（概率），predicted是索引（预测结果）
        # 遍历当前批次的每张图像，显示预测结果
        for i in range(images.size(0)):
            # 1. 处理图像格式：从张量转为OpenCV可显示的格式
            img = images[i].cpu().numpy().transpose(1, 2, 0)  # 从[C, H, W]转为[H, W, C]（OpenCV要求的格式）
            img = (img * 0.5 + 0.5) * 255  # 反归一化，从[-1,-1]到[0, 255]
            img = img.astype('uint8')      # 转为整数类型（像素值必须是0-255的整数）
            # 2. 显示图像，窗口标题为预测结果
            img_resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)  # 放大图像以便观察(Inter_LINEAR双线性插值,避免模糊)
            cv2.imshow(f'Predicted: {predicted[i].item()}', img_resized)
            # 3. 按任意键显示下一张，按q键直接退出所有窗口
            key = cv2.waitKey(0)  # 等待按键
            if key & 0xFF == ord('q'):  # 按q退出
                cv2.destroyAllWindows() # 关闭所有窗口
                exit()  # 退出程序
            else:
                cv2.destroyWindow(f'Predicted: {predicted[i].item()}') # 关闭当前窗口
        cv2.destroyAllWindows() # 处理完一个批次后关闭所有窗口
        break  # 只处理一个批次以示例展示
