import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 超分辨率处理函数
def super_resolve(image_path, output_dir='results', model_name='esrgan'):
    """
    使用预训练模型对图像进行超分辨率处理
    
    参数：
        image_path ("C:\\Users\\20858\\Desktop\\kcf9.com-(2).jpg"): 输入图片路径
        output_dir (str): 输出目录（默认'results'）
        model_name (str): 使用的模型（目前支持'esrgan'）
    """
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 加载预训练模型
        if model_name == 'esrgan':
           model = torch.hub.load('AK391/animegan2-pytorch:main', 'generator', pretrained=True, device='cuda')
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.eval().to(device)
        
        # 加载并预处理图像
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 执行超分辨率
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 后处理
        output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
        output_img = transforms.ToPILImage()(output_tensor)

        # 保存结果
        base_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"SR_{base_name}")
        output_img.save(save_path)
        print(f"Successfully saved super-resolved image to: {save_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # 配置参数（用户可修改部分）
    input_image = "input.jpg"       # 输入图片路径
    output_directory = "output"      # 输出目录
    selected_model = "esrgan"        # 选择使用的模型

    # 执行超分辨率处理
    super_resolve(
        image_path="C:\\Users\\20858\\Downloads\\IMG_7177.PNG",
        output_dir="C:\\Users\\20858\\Desktop",
        model_name='esrgan'
    )