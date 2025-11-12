# 图像退化算法工具

基于AlbumentationsX的图像退化算法演示工具，提供图形界面来测试和预览各种图像退化效果。

## 功能特点

- 📸 支持33种不同的退化算法
- 🎛️ 实时参数调节和预览
- 👁️ 左右对比显示原图和效果图
- 💾 支持保存处理结果

## 支持的算法

### 图像压缩与质量退化
- ImageCompression - JPEG/WebP压缩
- Downscale - 下采样再上采样
- Posterize - 降低位深度

### 噪声退化
- GaussNoise - 高斯噪声
- ISONoise - ISO传感器噪声
- ShotNoise - 泊松噪声
- SaltAndPepper - 椒盐噪声
- MultiplicativeNoise - 乘性噪声
- AdditiveNoise - 加性噪声

### 模糊退化
- Blur - 均匀盒状模糊
- GaussianBlur - 高斯模糊
- MotionBlur - 运动模糊
- MedianBlur - 中值模糊
- GlassBlur - 玻璃模糊
- Defocus - 散焦模糊

### Dropout类退化
- CoarseDropout - 粗粒度区域丢弃
- GridDropout - 网格模式丢弃
- PixelDropout - 像素级丢弃
- ChannelDropout - 通道丢弃

### 几何畸变退化
- ElasticTransform - 弹性变形
- GridDistortion - 网格畸变
- OpticalDistortion - 光学畸变（镜头畸变）
- PiecewiseAffine - 分段仿射变形
- ThinPlateSpline - 薄板样条变形

### 其他退化
- Dithering - 抖动算法
- RingingOvershoot - 振铃伪影
- ChromaticAberration - 色差

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python degradation_gui.py
```

1. 点击"选择图片"加载图片
2. 在下拉菜单选择算法
3. 使用滑块调整参数
4. 点击"应用算法"查看效果
5. 点击"保存结果"保存处理后的图片

## 项目结构

```
img_degrade/
├── albumentations/      # 核心退化算法实现
├── degradation_gui.py   # GUI界面
└── requirements.txt     # 依赖文件
```

