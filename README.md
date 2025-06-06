基于opencv的织物纺织缺陷的识别

主要利用opencv库实现对织物表面缺陷的简单识别，通过该程序可以识别织物表面的常见缺陷

其流程主要是灰度化-->傅里叶变换-->高通滤波-->傅里叶逆变换-->边缘增强-->最小值滤波-->二值化-->中值滤波
![image](https://github.com/user-attachments/assets/c694074a-6310-45f1-b94e-17e451d4fb57)

上述代码有综合和分步骤两部分。

不足：傅里叶变换在信号频域分析中有着重要作用，但它只能对整个时间段的信号的频率进行分析，没有信号的空间局部信息的刻画能力，如当需要对局部的图像纹理细节进行分析时，傅里叶变换无能为力。在进行阈值操作将图像二值化时，需要不断手动调节阈值获得最优的二值化图像，同时在进行最小值滤波和中值滤波时，也需要手动调节滤波器的大小从而获得最优的滤波后的图像。

改进思路：可以通过改进滤波器与算子，获得更好的图像缺陷检测。或者利用主流的深度学习方法，对含有缺陷图像的数据集中的图片进行训练，得到拟合能力和泛化能力较好的模型，从而对缺陷进行自动检测。



