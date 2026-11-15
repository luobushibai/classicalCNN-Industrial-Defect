# Classical CNN for Industrial Defect Classification

基于 MobileNetV2 倒残差骨架，引入 Gabor 纹理卷积与 TEAttention 注意力机制，用于 NEU-CLS 钢材表面缺陷分类。

## 实验成果
- **模型参数量**：2.301 M
- **计算量 (FLOPs)**：90.301 M
- **完整模型 (含TEA)**：Top-1 Acc **96.39%**
- **消融实验 (无TEA)**：Top-1 Acc **93.61%**

## 数据集下载 (Datasets)
1. **主实验数据集 (NEU-CLS 钢材表面缺陷)**:
   [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm)
2. **跨场景挑战数据集 (MVTec LOCO AD)**:
   [MVTec LOCO AD Downloads](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad/downloads)

## 运行步骤 (Usage)
1. 准备数据集并在 `model_data/neu_cls.txt` 中配置类别。
2. 生成数据标签：`python txt_annotation.py`
3. 测算参数量：`python summary.py`
4. 模型训练：`python train.py`
5. 测试集评估：`python eval.py`
EOF
