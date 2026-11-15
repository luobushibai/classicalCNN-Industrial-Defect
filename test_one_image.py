import torch
from PIL import Image
import numpy as np
from classification import Classification
from utils.utils import preprocess_input, letterbox_image

if __name__ == "__main__":
    # 1. 加载模型（这里会自动读取 eval.py 里设置的最优权重）
    # 注意：确保你之前跑的 eval.py 是对的
    model = Classification(
        classes_path='model_data/neu_cls.txt',
        model_path='logs/best_epoch_weights.pth'
    )

    # 2. 输入你要测试的图片路径
    image_path = 'test_image_2.bmp'  # <--- 改成你上传的图片名字
    
    print(f"\n正在预测图片: {image_path}")
    
    # 3. 读取并预处理图片
    # 强制把图片转为 RGB 三通道
    image = Image.open(image_path).convert('RGB')
    image_data = letterbox_image(image, [224, 224], False)
    image_data = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

    # 4. 送入模型预测
    with torch.no_grad():
        photo = torch.from_numpy(image_data)
        if torch.cuda.is_available():
            photo = photo.cuda()
        
        # 获取 6 个类别的概率
        preds = torch.softmax(model.model(photo)[0], dim=-1).cpu().numpy()

    # 5. 输出结果
    class_names = model.class_names
    # 按概率从高到低排序
    sorted_indices = np.argsort(preds)[::-1]
    
    print("-" * 30)
    print("预测结果排名：")
    for i in sorted_indices:
        print(f"{class_names[i]:<20}: {preds[i]*100:.2f}%")
    print("-" * 30)
    print(f"最终判定为：{class_names[sorted_indices[0]]}")