import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 定义彩色版本的SRCNN模型（输入和输出均为3通道）
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 图片加载函数（加载彩色图片）
def load_image(path, scale_factor=2, max_size=400):
    image = Image.open(path).convert('RGB')  # 加载彩色图像
    # 调整图片尺寸（可选）
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    # 使用ToTensor将图片转换为张量（归一化到[0,1]）
    hr_transform = transforms.ToTensor()
    hr_image = hr_transform(image).unsqueeze(0)  # 高分辨率图像

    # 制造低分辨率图像：先下采样再上采样
    lr_image = image.resize((image.size[0] // scale_factor, image.size[1] // scale_factor), Image.BICUBIC)
    lr_image = lr_image.resize((image.size[0], image.size[1]), Image.BICUBIC)
    lr_image = transforms.ToTensor()(lr_image).unsqueeze(0)
    return hr_image, lr_image

# 请替换下面的路径为你本地图片的实际路径（使用原始字符串或双反斜杠）
image_path = r"C:\\Users\\20858\\Desktop\\kcf9.com-(2).jpg"
hr_image, lr_image = load_image(image_path, scale_factor=2)
hr_image = hr_image.to(device)
lr_image = lr_image.to(device)

# 训练（这里仅训练50个epoch作为示例）
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(lr_image)
    loss = criterion(outputs, hr_image)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试阶段：生成超分辨率图像
model.eval()
with torch.no_grad():
    sr_image = model(lr_image)

# 辅助函数：将tensor转换为可用于imshow显示的numpy数组
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach().numpy().squeeze()
    # tensor形状为 (C, H, W)，转换为 (H, W, C)
    image = image.transpose(1, 2, 0)
    # 将值限制在0~1之间
    image = np.clip(image, 0, 1)
    return image

# 显示低分辨率、SRCNN输出和高分辨率图片
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Low Resolution")
plt.imshow(im_convert(lr_image))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("SRCNN Output")
plt.imshow(im_convert(sr_image))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("High Resolution")
plt.imshow(im_convert(hr_image))
plt.axis('off')

plt.show()

# 保存生成的超分辨率图像到本地
# def save_image(tensor, path):
#     image = tensor.to("cpu").clone().detach().squeeze(0)  # 去除batch维度
#     image = transforms.ToPILImage()(image)
#     image.save(path)
# 修改后的保存函数（添加数值归一化处理）
def save_image(tensor, path):
    # 将数值限制到合理范围
    tensor = torch.clamp(tensor, 0.0, 1.0)  # 新增：强制限制数值范围
    image = tensor.to("cpu").clone().detach().squeeze(0)
    image = transforms.ToPILImage()(image)
    
    # 添加锐化处理（可选）
    #image = image.filter(ImageFilter.SHARPEN) if sharpen else image
    
    image.save(path, quality=95)  # 提高保存质量参数

save_image(sr_image, "output_super_resolution.jpg")
print("Super-resolution image saved as output_super_resolution.jpg")
