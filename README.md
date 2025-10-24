Introduction to Deep Learnong:Implementing Digit Recognition with Custom Network Models.

models.py是自定义神经网络模型构建文件，使用全连接
train1.py包含数据处理与加载，数据集为pytorch自带的MNIST数据集，是模型训练文件
predict.py加载MNIST测试集中的图片进行预测
predict1.py加载本地图片进行推理，需注意，图片预处理，以及输入图像的特征必须与训练集一致（黑底白字，像素为28*28），myfigure.png没有调整像素大小，使用强制压缩图片到28*28，推理结果错误，myfigure2.jpg手动裁剪过了，预测正确
test1.ipynb是一个jupyter文件，可以自行测试代码
training_curves.png是训练结果的可视化评估
