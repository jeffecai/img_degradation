#!/usr/bin/env python3
"""
图像退化算法GUI工具
支持选择图片、选择退化算法、调整参数并实时预览效果
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import albumentations as A
from typing import Any


class DegradationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("图像退化算法演示工具")
        self.root.geometry("1400x900")
        
        # 存储当前图片
        self.original_image = None
        self.original_image_rgb = None
        self.current_result = None
        self.current_image_path = None  # 存储当前图片路径
        self.range_param_vars = {}  # 存储参数范围输入框的变量
        
        # 创建界面
        self.create_widgets()
        
        # 算法配置
        self.setup_algorithms()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.root, padding="10")
        toolbar.pack(fill=tk.X)
        
        ttk.Button(toolbar, text="选择图片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="保存结果", command=self.save_result).pack(side=tk.LEFT, padx=5)
        
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：控制面板
        control_panel = ttk.LabelFrame(main_container, text="算法控制", padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.config(width=350)
        
        # 算法选择
        ttk.Label(control_panel, text="选择退化算法:").pack(anchor=tk.W, pady=(0, 5))
        self.algorithm_var = tk.StringVar(value="ImageCompression")
        self.algorithm_combo = ttk.Combobox(
            control_panel, 
            textvariable=self.algorithm_var,
            state="readonly",
            width=35
        )
        self.algorithm_combo.pack(fill=tk.X, pady=(0, 10))
        self.algorithm_combo.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        # 算法说明区域
        self.algorithm_desc_frame = ttk.LabelFrame(control_panel, text="算法说明", padding="5")
        self.algorithm_desc_frame.pack(fill=tk.X, pady=(0, 10))
        self.algorithm_desc_text = tk.Text(
            self.algorithm_desc_frame,
            height=3,
            wrap=tk.WORD,
            font=("Arial", 9),
            state=tk.DISABLED
        )
        self.algorithm_desc_text.pack(fill=tk.BOTH, expand=True)
        
        # 参数控制区域
        self.param_frame = ttk.Frame(control_panel)
        self.param_frame.pack(fill=tk.BOTH, expand=True)
        
        # 应用按钮
        ttk.Button(
            control_panel, 
            text="应用算法", 
            command=self.apply_algorithm
        ).pack(fill=tk.X, pady=(10, 0))
        
        # 批量生成区域
        batch_frame = ttk.LabelFrame(control_panel, text="批量生成随机样本", padding="5")
        batch_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 样本数量输入
        ttk.Label(batch_frame, text="样本数量:").pack(anchor=tk.W, pady=(0, 2))
        self.sample_count_var = tk.StringVar(value="10")
        sample_count_entry = ttk.Entry(batch_frame, textvariable=self.sample_count_var, width=15)
        sample_count_entry.pack(fill=tk.X, pady=(0, 5))
        
        # 参数范围设置区域（滚动区域）
        range_scroll_frame = ttk.Frame(batch_frame)
        range_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动条
        range_canvas = tk.Canvas(range_scroll_frame, height=150)
        range_scrollbar = ttk.Scrollbar(range_scroll_frame, orient="vertical", command=range_canvas.yview)
        self.range_params_frame = ttk.Frame(range_canvas)
        
        self.range_params_frame.bind(
            "<Configure>",
            lambda e: range_canvas.configure(scrollregion=range_canvas.bbox("all"))
        )
        
        range_canvas.create_window((0, 0), window=self.range_params_frame, anchor="nw")
        range_canvas.configure(yscrollcommand=range_scrollbar.set)
        
        range_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        range_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 保存范围随机结果按钮
        ttk.Button(
            batch_frame, 
            text="保存范围随机结果", 
            command=self.save_random_samples
        ).pack(fill=tk.X, pady=(5, 0))
        
        # 右侧：图片显示区域
        display_frame = ttk.Frame(main_container)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 原图显示
        original_frame = ttk.LabelFrame(display_frame, text="原图", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, bg="gray", highlightthickness=1)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 效果图显示
        result_frame = ttk.LabelFrame(display_frame, text="效果图", padding="5")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(result_frame, bg="gray", highlightthickness=1)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
    def setup_algorithms(self):
        """设置算法列表和参数配置"""
        # 算法中英文名称映射
        self.algorithm_names = {
            "ImageCompression": "图像压缩 / Image Compression",
            "Downscale": "下采样 / Downscale",
            "GaussNoise": "高斯噪声 / Gaussian Noise",
            "Blur": "均匀模糊 / Blur",
            "GaussianBlur": "高斯模糊 / Gaussian Blur",
            "MotionBlur": "运动模糊 / Motion Blur",
            "MedianBlur": "中值模糊 / Median Blur",
            "ISONoise": "ISO噪声 / ISO Noise",
            "ShotNoise": "泊松噪声 / Shot Noise",
            "SaltAndPepper": "椒盐噪声 / Salt and Pepper",
            "Posterize": "色调分离 / Posterize",
            "Dithering": "抖动 / Dithering",
            "RingingOvershoot": "振铃伪影 / Ringing Overshoot",
            "MultiplicativeNoise": "乘性噪声 / Multiplicative Noise",
            "AdditiveNoise": "加性噪声 / Additive Noise",
            "Defocus": "散焦 / Defocus",
            "GlassBlur": "玻璃模糊 / Glass Blur",
            "ChromaticAberration": "色差 / Chromatic Aberration",
            "CoarseDropout": "粗粒度丢弃 / Coarse Dropout",
            "GridDropout": "网格丢弃 / Grid Dropout",
            "PixelDropout": "像素丢弃 / Pixel Dropout",
            "ChannelDropout": "通道丢弃 / Channel Dropout",
            "ElasticTransform": "弹性变形 / Elastic Transform",
            "GridDistortion": "网格畸变 / Grid Distortion",
            "OpticalDistortion": "光学畸变 / Optical Distortion",
            "PiecewiseAffine": "分段仿射 / Piecewise Affine",
            "ThinPlateSpline": "薄板样条 / Thin Plate Spline"
        }
        
        # 参数说明字典
        self.param_descriptions = {
            "quality_range": {
                "name": "质量范围",
                "description": "控制JPEG/WebP压缩质量。值越小压缩越强，图像质量越低，文件越小。值越大质量越高，但文件也越大。",
                "effect_up": "值增大 → 图像质量提高，压缩伪影减少",
                "effect_down": "值减小 → 图像质量降低，压缩伪影增多（块状、模糊）"
            },
            "compression_type": {
                "name": "压缩类型",
                "description": "选择压缩算法类型。JPEG适合照片，WebP压缩率更高但兼容性稍差。",
                "effect_up": "",
                "effect_down": ""
            },
            "scale_range": {
                "name": "缩放范围",
                "description": "控制下采样比例。值越小，图像先缩小得越多再放大回来，细节损失越大。",
                "effect_up": "值增大 → 细节保留更多，图像更清晰",
                "effect_down": "值减小 → 细节损失更多，图像更模糊（像素化）"
            },
            "std_range": {
                "name": "标准差范围",
                "description": "控制高斯噪声的强度。值越大，噪声越明显，图像越粗糙。",
                "effect_up": "值增大 → 噪声增强，图像更粗糙",
                "effect_down": "值减小 → 噪声减弱，图像更平滑"
            },
            "mean_range": {
                "name": "均值范围",
                "description": "控制噪声的偏移量。正值使图像整体变亮，负值使图像整体变暗。",
                "effect_up": "值增大 → 图像整体变亮",
                "effect_down": "值减小 → 图像整体变暗"
            },
            "per_channel": {
                "name": "每通道独立",
                "description": "是否对每个颜色通道（R、G、B）独立添加噪声。开启后颜色可能产生偏移。",
                "effect_up": "",
                "effect_down": ""
            },
            "blur_limit": {
                "name": "模糊强度",
                "description": "控制模糊核的大小。值越大，模糊效果越强，图像越不清晰。必须是奇数（3, 5, 7...）。",
                "effect_up": "值增大 → 模糊增强，图像更不清晰",
                "effect_down": "值减小 → 模糊减弱，图像更清晰"
            },
            "sigma_limit": {
                "name": "Sigma范围",
                "description": "控制高斯模糊的标准差。值越大，模糊范围越大，过渡越平滑。",
                "effect_up": "值增大 → 模糊范围扩大，过渡更平滑",
                "effect_down": "值减小 → 模糊范围缩小，过渡更锐利"
            },
            "allow_shifted": {
                "name": "允许偏移",
                "description": "是否允许运动模糊方向随机偏移。开启后运动方向更随机。",
                "effect_up": "",
                "effect_down": ""
            },
            "color_shift": {
                "name": "颜色偏移",
                "description": "控制ISO噪声引起的颜色偏移程度。值越大，颜色失真越明显。",
                "effect_up": "值增大 → 颜色偏移增强，色彩失真更明显",
                "effect_down": "值减小 → 颜色偏移减弱，色彩更准确"
            },
            "intensity": {
                "name": "强度",
                "description": "控制ISO噪声的整体强度。值越大，噪声越明显。",
                "effect_up": "值增大 → 噪声增强，图像质量下降",
                "effect_down": "值减小 → 噪声减弱，图像质量提高"
            },
            "scale_range": {
                "name": "噪声强度",
                "description": "控制泊松噪声的强度。值越大，噪声越明显。",
                "effect_up": "值增大 → 噪声增强",
                "effect_down": "值减小 → 噪声减弱"
            },
            "amount": {
                "name": "噪声量",
                "description": "控制椒盐噪声的像素比例。值越大，被噪声点替换的像素越多。",
                "effect_up": "值增大 → 噪声点增多，图像更粗糙",
                "effect_down": "值减小 → 噪声点减少，图像更干净"
            },
            "salt_vs_pepper": {
                "name": "盐/椒比例",
                "description": "控制白色噪声点（盐）和黑色噪声点（椒）的比例。0.5表示各占一半。",
                "effect_up": "值增大 → 白色噪声点增多",
                "effect_down": "值减小 → 黑色噪声点增多"
            },
            "num_bits": {
                "name": "位数",
                "description": "控制每个颜色通道保留的位数。值越小，颜色层次越少，图像越像海报。",
                "effect_up": "值增大 → 颜色层次增多，图像更细腻",
                "effect_down": "值减小 → 颜色层次减少，图像更粗糙（海报化）"
            },
            "method": {
                "name": "抖动方法",
                "description": "选择抖动算法。random=随机噪声，ordered=有序网格，error_diffusion=误差扩散（质量最好）。",
                "effect_up": "",
                "effect_down": ""
            },
            "n_colors": {
                "name": "颜色数",
                "description": "控制每个通道的颜色数量。值越小，颜色越少，抖动效果越明显。",
                "effect_up": "值增大 → 颜色增多，抖动效果减弱",
                "effect_down": "值减小 → 颜色减少，抖动效果增强（复古感）"
            },
            "color_mode": {
                "name": "颜色模式",
                "description": "per_channel=每通道独立抖动，grayscale=先转灰度再抖动。",
                "effect_up": "",
                "effect_down": ""
            },
            "error_diffusion_algorithm": {
                "name": "误差扩散算法",
                "description": "选择误差扩散的具体算法。Floyd-Steinberg最常用，Jarvis质量最高但较慢。",
                "effect_up": "",
                "effect_down": ""
            },
            "cutoff": {
                "name": "截止频率",
                "description": "控制振铃伪影的频率。值越大，振铃效果越明显，边缘振荡越强。",
                "effect_up": "值增大 → 振铃效果增强，边缘振荡更明显",
                "effect_down": "值减小 → 振铃效果减弱，边缘更平滑"
            },
            "multiplier": {
                "name": "乘数范围",
                "description": "控制乘性噪声的强度。值偏离1.0越多，噪声越明显。",
                "effect_up": "值增大 → 噪声增强",
                "effect_down": "值减小 → 噪声减弱"
            },
            "noise_type": {
                "name": "噪声类型",
                "description": "选择噪声分布类型。uniform=均匀分布，gaussian=高斯分布，laplace=拉普拉斯分布。",
                "effect_up": "",
                "effect_down": ""
            },
            "spatial_mode": {
                "name": "空间模式",
                "description": "constant=每通道一个值，per_pixel=每像素独立，shared=所有通道共享。",
                "effect_up": "",
                "effect_down": ""
            },
            "radius": {
                "name": "半径",
                "description": "控制散焦模糊的半径。值越大，模糊范围越大。",
                "effect_up": "值增大 → 模糊范围扩大",
                "effect_down": "值减小 → 模糊范围缩小"
            },
            "alias_blur": {
                "name": "别名模糊",
                "description": "控制散焦的别名模糊程度。值越大，模糊效果越平滑。",
                "effect_up": "值增大 → 模糊更平滑",
                "effect_down": "值减小 → 模糊更锐利"
            },
            "sigma": {
                "name": "Sigma",
                "description": "控制玻璃模糊的标准差。值越大，模糊效果越强。",
                "effect_up": "值增大 → 模糊增强",
                "effect_down": "值减小 → 模糊减弱"
            },
            "max_delta": {
                "name": "最大偏移",
                "description": "控制玻璃模糊的最大像素偏移。值越大，扭曲效果越明显。",
                "effect_up": "值增大 → 扭曲增强",
                "effect_down": "值减小 → 扭曲减弱"
            },
            "iterations": {
                "name": "迭代次数",
                "description": "控制玻璃模糊的迭代次数。值越大，效果越强但计算越慢。",
                "effect_up": "值增大 → 效果增强但速度变慢",
                "effect_down": "值减小 → 效果减弱但速度变快"
            },
            "shift_x": {
                "name": "X偏移",
                "description": "控制色差在X方向的偏移。正值向右偏移，负值向左偏移。",
                "effect_up": "值增大 → 向右偏移增强",
                "effect_down": "值减小 → 向左偏移增强"
            },
            "shift_y": {
                "name": "Y偏移",
                "description": "控制色差在Y方向的偏移。正值向下偏移，负值向上偏移。",
                "effect_up": "值增大 → 向下偏移增强",
                "effect_down": "值减小 → 向上偏移增强"
            },
            "num_holes_range": {
                "name": "空洞数量",
                "description": "控制丢弃的矩形区域数量。值越大，被遮挡的区域越多。",
                "effect_up": "值增大 → 遮挡区域增多",
                "effect_down": "值减小 → 遮挡区域减少"
            },
            "hole_height_range": {
                "name": "空洞高度(比例)",
                "description": "控制丢弃区域的高度（相对于图像高度的比例）。值越大，空洞越高。",
                "effect_up": "值增大 → 空洞高度增加",
                "effect_down": "值减小 → 空洞高度减少"
            },
            "hole_width_range": {
                "name": "空洞宽度(比例)",
                "description": "控制丢弃区域的宽度（相对于图像宽度的比例）。值越大，空洞越宽。",
                "effect_up": "值增大 → 空洞宽度增加",
                "effect_down": "值减小 → 空洞宽度减少"
            },
            "ratio": {
                "name": "空洞比例",
                "description": "控制网格中每个单元被丢弃的比例。值越大，丢弃的区域越多。",
                "effect_up": "值增大 → 丢弃区域增多",
                "effect_down": "值减小 → 丢弃区域减少"
            },
            "unit_size_range": {
                "name": "网格单元大小",
                "description": "控制网格单元的大小。值越大，网格越大，丢弃区域越大。",
                "effect_up": "值增大 → 网格变大，丢弃区域变大",
                "effect_down": "值减小 → 网格变小，丢弃区域变小"
            },
            "random_offset": {
                "name": "随机偏移",
                "description": "是否随机偏移网格位置。开启后网格位置更随机。",
                "effect_up": "",
                "effect_down": ""
            },
            "dropout_prob": {
                "name": "像素丢弃概率",
                "description": "控制每个像素被丢弃的概率。值越大，被丢弃的像素越多。",
                "effect_up": "值增大 → 丢弃像素增多",
                "effect_down": "值减小 → 丢弃像素减少"
            },
            "channel_drop_range": {
                "name": "丢弃通道数",
                "description": "控制丢弃的颜色通道数量。值越大，可能丢弃的通道越多。",
                "effect_up": "值增大 → 可能丢弃更多通道",
                "effect_down": "值减小 → 可能丢弃更少通道"
            },
            "fill": {
                "name": "填充值",
                "description": "控制丢弃区域的填充值。0=黑色，255=白色。",
                "effect_up": "值增大 → 填充更亮",
                "effect_down": "值减小 → 填充更暗"
            },
            "alpha": {
                "name": "弹性强度",
                "description": "控制弹性变形的强度。值越大，变形越明显。",
                "effect_up": "值增大 → 变形增强，扭曲更明显",
                "effect_down": "值减小 → 变形减弱，扭曲更轻微"
            },
            "sigma": {
                "name": "平滑度",
                "description": "控制弹性变形的平滑程度。值越大，变形越平滑。",
                "effect_up": "值增大 → 变形更平滑",
                "effect_down": "值减小 → 变形更锐利"
            },
            "num_steps": {
                "name": "网格步数",
                "description": "控制网格畸变的网格密度。值越大，网格越密，畸变越精细。",
                "effect_up": "值增大 → 网格更密，畸变更精细",
                "effect_down": "值减小 → 网格更疏，畸变更粗糙"
            },
            "distort_limit": {
                "name": "畸变强度",
                "description": "控制畸变的强度。值可以是正负，绝对值越大，图像扭曲越明显。",
                "effect_up": "绝对值增大 → 扭曲增强",
                "effect_down": "绝对值减小 → 扭曲减弱"
            },
            "scale": {
                "name": "变形强度",
                "description": "控制变形的强度。值越大，变形越明显。",
                "effect_up": "值增大 → 变形增强",
                "effect_down": "值减小 → 变形减弱"
            },
            "scale_range": {
                "name": "变形强度范围",
                "description": "控制薄板样条变形的强度范围（[0,1]）。值越大，变形越明显。",
                "effect_up": "值增大 → 变形增强",
                "effect_down": "值减小 → 变形减弱"
            },
            "nb_rows": {
                "name": "网格行数",
                "description": "控制分段仿射的网格行数。值越大，网格越密，变形越精细。",
                "effect_up": "值增大 → 网格更密，变形更精细",
                "effect_down": "值减小 → 网格更疏，变形更粗糙"
            },
            "nb_cols": {
                "name": "网格列数",
                "description": "控制分段仿射的网格列数。值越大，网格越密，变形越精细。",
                "effect_up": "值增大 → 网格更密，变形更精细",
                "effect_down": "值减小 → 网格更疏，变形更粗糙"
            },
            "num_control_points": {
                "name": "控制点数",
                "description": "控制薄板样条的控制点数量（每边）。值越大，变形越精细。",
                "effect_up": "值增大 → 变形更精细",
                "effect_down": "值减小 → 变形更粗糙"
            }
        }
        
        self.algorithms = {
            "ImageCompression": {
                "class": A.ImageCompression,
                "params": {
                    "quality_range": {"type": "range_int", "min": 1, "max": 100, "default": (50, 80), "label": "质量范围"},
                    "compression_type": {"type": "choice", "options": ["jpeg", "webp"], "default": "jpeg", "label": "压缩类型"}
                }
            },
            "Downscale": {
                "class": A.Downscale,
                "params": {
                    "scale_range": {"type": "range", "min": 0.1, "max": 1.0, "default": (0.3, 0.7), "label": "缩放范围"}
                }
            },
            "GaussNoise": {
                "class": A.GaussNoise,
                "params": {
                    "std_range": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.1, 0.3), "label": "标准差范围"},
                    "mean_range": {"type": "range", "min": -1.0, "max": 1.0, "default": (0.0, 0.0), "label": "均值范围"},
                    "per_channel": {"type": "bool", "default": False, "label": "每通道独立"}
                }
            },
            "Blur": {
                "class": A.Blur,
                "params": {
                    "blur_limit": {"type": "range_int", "min": 3, "max": 31, "default": (3, 15), "label": "模糊强度(奇数)"}
                }
            },
            "GaussianBlur": {
                "class": A.GaussianBlur,
                "params": {
                    "blur_limit": {"type": "range_int", "min": 3, "max": 31, "default": (3, 15), "label": "模糊强度(奇数)"},
                    "sigma_limit": {"type": "range", "min": 0.0, "max": 5.0, "default": (0.0, 3.0), "label": "Sigma范围"}
                }
            },
            "MotionBlur": {
                "class": A.MotionBlur,
                "params": {
                    "blur_limit": {"type": "range_int", "min": 3, "max": 31, "default": (3, 15), "label": "模糊强度(奇数)"},
                    "allow_shifted": {"type": "bool", "default": True, "label": "允许偏移"}
                }
            },
            "MedianBlur": {
                "class": A.MedianBlur,
                "params": {
                    "blur_limit": {"type": "range_int", "min": 3, "max": 31, "default": (3, 15), "label": "模糊强度(奇数)"}
                }
            },
            "ISONoise": {
                "class": A.ISONoise,
                "params": {
                    "color_shift": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.1, 0.3), "label": "颜色偏移"},
                    "intensity": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.5, 0.9), "label": "强度"}
                }
            },
            "ShotNoise": {
                "class": A.ShotNoise,
                "params": {
                    "scale_range": {"type": "range", "min": 0.01, "max": 2.0, "default": (0.1, 0.5), "label": "噪声强度"}
                }
            },
            "SaltAndPepper": {
                "class": A.SaltAndPepper,
                "params": {
                    "amount": {"type": "range", "min": 0.0, "max": 0.3, "default": (0.01, 0.05), "label": "噪声量"},
                    "salt_vs_pepper": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.4, 0.6), "label": "盐/椒比例"}
                }
            },
            "Posterize": {
                "class": A.Posterize,
                "params": {
                    "num_bits": {"type": "range_int", "min": 1, "max": 7, "default": (2, 5), "label": "位数"}
                }
            },
            "Dithering": {
                "class": A.Dithering,
                "params": {
                    "method": {"type": "choice", "options": ["random", "ordered", "error_diffusion"], "default": "error_diffusion", "label": "抖动方法"},
                    "n_colors": {"type": "range_int", "min": 2, "max": 256, "default": (2, 8), "label": "颜色数"},
                    "color_mode": {"type": "choice", "options": ["per_channel", "grayscale"], "default": "grayscale", "label": "颜色模式"},
                    "error_diffusion_algorithm": {"type": "choice", "options": ["floyd_steinberg", "jarvis", "stucki", "atkinson", "burkes", "sierra", "sierra_2row", "sierra_lite"], "default": "floyd_steinberg", "label": "误差扩散算法"}
                }
            },
            "RingingOvershoot": {
                "class": A.RingingOvershoot,
                "params": {
                    "blur_limit": {"type": "range_int", "min": 3, "max": 31, "default": (7, 15), "label": "核大小(奇数)"},
                    "cutoff": {"type": "range", "min": 0.1, "max": 3.14, "default": (0.785, 1.57), "label": "截止频率"}
                }
            },
            "MultiplicativeNoise": {
                "class": A.MultiplicativeNoise,
                "params": {
                    "multiplier": {"type": "range", "min": 0.5, "max": 1.5, "default": (0.8, 1.2), "label": "乘数范围"},
                    "per_channel": {"type": "bool", "default": False, "label": "每通道独立"}
                }
            },
            "AdditiveNoise": {
                "class": A.AdditiveNoise,
                "params": {
                    "noise_type": {"type": "choice", "options": ["uniform", "gaussian", "laplace"], "default": "gaussian", "label": "噪声类型"},
                    "spatial_mode": {"type": "choice", "options": ["constant", "per_pixel", "shared"], "default": "per_pixel", "label": "空间模式"}
                },
                "custom_init": True
            },
            "Defocus": {
                "class": A.Defocus,
                "params": {
                    "radius": {"type": "range_int", "min": 1, "max": 10, "default": (1, 5), "label": "半径"},
                    "alias_blur": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.1, 0.3), "label": "别名模糊"}
                }
            },
            "GlassBlur": {
                "class": A.GlassBlur,
                "params": {
                    "sigma": {"type": "range", "min": 0.0, "max": 5.0, "default": (0.5, 2.0), "label": "Sigma"},
                    "max_delta": {"type": "range_int", "min": 1, "max": 10, "default": (1, 5), "label": "最大偏移"},
                    "iterations": {"type": "range_int", "min": 1, "max": 5, "default": (1, 2), "label": "迭代次数"}
                }
            },
            "ChromaticAberration": {
                "class": A.ChromaticAberration,
                "params": {
                    "shift_x": {"type": "range_int", "min": -10, "max": 10, "default": (-5, 5), "label": "X偏移"},
                    "shift_y": {"type": "range_int", "min": -10, "max": 10, "default": (-5, 5), "label": "Y偏移"}
                }
            },
            # Dropout类退化算法
            "CoarseDropout": {
                "class": A.CoarseDropout,
                "params": {
                    "num_holes_range": {"type": "range_int", "min": 1, "max": 10, "default": (1, 5), "label": "空洞数量"},
                    "hole_height_range": {"type": "range", "min": 0.05, "max": 0.5, "default": (0.1, 0.2), "label": "空洞高度(比例)"},
                    "hole_width_range": {"type": "range", "min": 0.05, "max": 0.5, "default": (0.1, 0.2), "label": "空洞宽度(比例)"}
                }
            },
            "GridDropout": {
                "class": A.GridDropout,
                "params": {
                    "ratio": {"type": "range", "min": 0.1, "max": 0.9, "default": (0.3, 0.6), "label": "空洞比例"},
                    "unit_size_range": {"type": "range_int", "min": 5, "max": 50, "default": (10, 30), "label": "网格单元大小"},
                    "random_offset": {"type": "bool", "default": True, "label": "随机偏移"}
                }
            },
            "PixelDropout": {
                "class": A.PixelDropout,
                "params": {
                    "dropout_prob": {"type": "range", "min": 0.01, "max": 0.3, "default": (0.01, 0.1), "label": "像素丢弃概率"},
                    "per_channel": {"type": "bool", "default": False, "label": "每通道独立"}
                }
            },
            "ChannelDropout": {
                "class": A.ChannelDropout,
                "params": {
                    "channel_drop_range": {"type": "range_int", "min": 1, "max": 3, "default": (1, 2), "label": "丢弃通道数"},
                    "fill": {"type": "range", "min": 0.0, "max": 255.0, "default": (0.0, 0.0), "label": "填充值"}
                }
            },
            # 几何畸变类退化算法
            "ElasticTransform": {
                "class": A.ElasticTransform,
                "params": {
                    "alpha": {"type": "range", "min": 0.0, "max": 200.0, "default": (1.0, 50.0), "label": "弹性强度"},
                    "sigma": {"type": "range", "min": 1.0, "max": 100.0, "default": (10.0, 50.0), "label": "平滑度"}
                }
            },
            "GridDistortion": {
                "class": A.GridDistortion,
                "params": {
                    "num_steps": {"type": "range_int", "min": 1, "max": 15, "default": (5, 10), "label": "网格步数"},
                    "distort_limit": {"type": "range", "min": -1.0, "max": 1.0, "default": (-0.3, 0.3), "label": "畸变强度"}
                }
            },
            "OpticalDistortion": {
                "class": A.OpticalDistortion,
                "params": {
                    "distort_limit": {"type": "range", "min": -0.3, "max": 0.3, "default": (-0.05, 0.05), "label": "畸变强度"}
                }
            },
            "PiecewiseAffine": {
                "class": A.PiecewiseAffine,
                "params": {
                    "scale": {"type": "range", "min": 0.01, "max": 0.1, "default": (0.03, 0.05), "label": "变形强度"},
                    "nb_rows": {"type": "range_int", "min": 2, "max": 10, "default": (4, 6), "label": "网格行数"},
                    "nb_cols": {"type": "range_int", "min": 2, "max": 10, "default": (4, 6), "label": "网格列数"}
                }
            },
            "ThinPlateSpline": {
                "class": A.ThinPlateSpline,
                "params": {
                    "scale_range": {"type": "range", "min": 0.0, "max": 1.0, "default": (0.2, 0.4), "label": "变形强度"},
                    "num_control_points": {"type": "range_int", "min": 2, "max": 10, "default": (3, 5), "label": "控制点数"}
                }
            }
        }
        
        # 更新算法下拉列表（显示中英文名称）
        algorithm_display_names = [self.algorithm_names.get(name, name) for name in self.algorithms.keys()]
        self.algorithm_combo["values"] = algorithm_display_names
        # 设置当前值为第一个算法的显示名称
        if algorithm_display_names:
            self.algorithm_combo.set(algorithm_display_names[0])
            # 找到对应的内部名称
            for internal_name, display_name in self.algorithm_names.items():
                if display_name == algorithm_display_names[0]:
                    self.algorithm_var.set(internal_name)
                    break
        self.on_algorithm_change()
        
    def on_algorithm_change(self, event=None):
        """当算法改变时更新参数控制"""
        # 获取选中的显示名称，转换为内部名称
        selected_display = self.algorithm_combo.get()
        algorithm_name = None
        for internal_name, display_name in self.algorithm_names.items():
            if display_name == selected_display:
                algorithm_name = internal_name
                self.algorithm_var.set(internal_name)
                break
        
        if algorithm_name is None:
            algorithm_name = self.algorithm_var.get()
        
        if algorithm_name not in self.algorithms:
            return
        
        # 更新算法说明
        self.update_algorithm_description(algorithm_name)
        
        # 清除旧的参数控件
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        # 清除旧的参数范围控件
        for widget in self.range_params_frame.winfo_children():
            widget.destroy()
        self.range_param_vars = {}
            
        algorithm_config = self.algorithms[algorithm_name]
        self.param_vars = {}
        
        # 创建参数范围输入控件
        self.create_range_inputs(algorithm_config)
        
        # 创建参数控件
        for param_name, param_config in algorithm_config["params"].items():
            param_type = param_config["type"]
            param_label = param_config.get("label", param_name)
            param_default = param_config.get("default")
            
            # 获取参数说明
            param_desc = self.param_descriptions.get(param_name, {})
            param_desc_name = param_desc.get("name", param_label)
            param_desc_text = param_desc.get("description", "")
            param_effect_up = param_desc.get("effect_up", "")
            param_effect_down = param_desc.get("effect_down", "")
            
            # 参数标签和说明
            label_frame = ttk.Frame(self.param_frame)
            label_frame.pack(fill=tk.X, pady=(10, 2))
            ttk.Label(label_frame, text=param_desc_name + ":", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
            
            # 参数说明文本
            if param_desc_text:
                desc_frame = ttk.Frame(self.param_frame)
                desc_frame.pack(fill=tk.X, pady=(0, 5))
                desc_label = ttk.Label(
                    desc_frame,
                    text=param_desc_text,
                    font=("Arial", 8),
                    foreground="gray",
                    wraplength=320,
                    justify=tk.LEFT
                )
                desc_label.pack(anchor=tk.W, padx=(10, 0))
                
                # 效果说明
                if param_effect_up or param_effect_down:
                    effect_text = ""
                    if param_effect_up:
                        effect_text += "↑ " + param_effect_up
                    if param_effect_down:
                        if effect_text:
                            effect_text += "\n"
                        effect_text += "↓ " + param_effect_down
                    
                    effect_label = ttk.Label(
                        desc_frame,
                        text=effect_text,
                        font=("Arial", 8),
                        foreground="blue",
                        wraplength=320,
                        justify=tk.LEFT
                    )
                    effect_label.pack(anchor=tk.W, padx=(10, 0), pady=(2, 0))
            
            if param_type == "range":
                # 固定值滑块（简化版，只显示单个值）
                default_value = param_default[0] if isinstance(param_default, tuple) else param_default
                # 如果默认是范围，取中间值
                if isinstance(param_default, tuple) and param_default[0] != param_default[1]:
                    default_value = (param_default[0] + param_default[1]) / 2.0
                
                var_value = tk.DoubleVar(value=default_value)
                
                # 单个值滑块
                value_frame = ttk.Frame(self.param_frame)
                value_frame.pack(fill=tk.X, pady=2)
                ttk.Label(value_frame, text="  值:", width=10).pack(side=tk.LEFT)
                scale_value = ttk.Scale(
                    value_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_value,
                    orient=tk.HORIZONTAL
                )
                scale_value.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_value = ttk.Label(value_frame, textvariable=var_value, width=8)
                label_value.pack(side=tk.LEFT)
                
                self.param_vars[param_name] = var_value
                
            elif param_type == "range_int":
                # 固定值滑块（简化版，只显示单个值）
                default_value = param_default[0] if isinstance(param_default, tuple) else param_default
                # 如果默认是范围，取中间值
                if isinstance(param_default, tuple) and param_default[0] != param_default[1]:
                    default_value = int((param_default[0] + param_default[1]) / 2)
                
                var_value = tk.IntVar(value=int(default_value))
                
                # 单个值滑块
                value_frame = ttk.Frame(self.param_frame)
                value_frame.pack(fill=tk.X, pady=2)
                ttk.Label(value_frame, text="  值:", width=10).pack(side=tk.LEFT)
                scale_value = ttk.Scale(
                    value_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_value,
                    orient=tk.HORIZONTAL
                )
                scale_value.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_value = ttk.Label(value_frame, textvariable=var_value, width=8)
                label_value.pack(side=tk.LEFT)
                
                self.param_vars[param_name] = var_value
                
            elif param_type == "choice":
                # 下拉选择
                var = tk.StringVar(value=param_default)
                combo = ttk.Combobox(
                    self.param_frame,
                    textvariable=var,
                    values=param_config["options"],
                    state="readonly",
                    width=25
                )
                combo.pack(fill=tk.X, pady=2)
                self.param_vars[param_name] = var
                
            elif param_type == "bool":
                # 复选框
                var = tk.BooleanVar(value=param_default)
                check = ttk.Checkbutton(
                    self.param_frame,
                    text="启用",
                    variable=var
                )
                check.pack(anchor=tk.W, pady=2)
                self.param_vars[param_name] = var
    
    def create_range_inputs(self, algorithm_config):
        """为批量生成创建参数范围输入控件"""
        for param_name, param_config in algorithm_config["params"].items():
            param_type = param_config["type"]
            param_label = param_config.get("label", param_name)
            param_default = param_config.get("default")
            
            # 只对range和range_int类型创建范围输入
            if param_type in ["range", "range_int"]:
                # 获取默认值
                if isinstance(param_default, tuple):
                    default_min = param_default[0]
                    default_max = param_default[1]
                else:
                    default_min = param_default
                    default_max = param_default
                
                # 如果默认是单个值，设置一个合理的范围
                if default_min == default_max:
                    if param_type == "range_int":
                        default_min = max(param_config["min"], int(default_min) - 2)
                        default_max = min(param_config["max"], int(default_max) + 2)
                    else:
                        range_size = (param_config["max"] - param_config["min"]) * 0.1
                        default_min = max(param_config["min"], default_min - range_size)
                        default_max = min(param_config["max"], default_max + range_size)
                
                # 参数标签
                param_frame = ttk.Frame(self.range_params_frame)
                param_frame.pack(fill=tk.X, pady=2)
                
                ttk.Label(param_frame, text=param_label + ":", width=15).pack(side=tk.LEFT, padx=(0, 5))
                
                # 最小值输入
                ttk.Label(param_frame, text="最小:", width=5).pack(side=tk.LEFT)
                var_min = tk.StringVar(value=str(default_min))
                entry_min = ttk.Entry(param_frame, textvariable=var_min, width=8)
                entry_min.pack(side=tk.LEFT, padx=2)
                
                # 最大值输入
                ttk.Label(param_frame, text="最大:", width=5).pack(side=tk.LEFT)
                var_max = tk.StringVar(value=str(default_max))
                entry_max = ttk.Entry(param_frame, textvariable=var_max, width=8)
                entry_max.pack(side=tk.LEFT, padx=2)
                
                self.range_param_vars[param_name] = {
                    "min": var_min,
                    "max": var_max,
                    "type": param_type
                }
    
    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # 使用OpenCV读取（BGR格式）
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                messagebox.showerror("错误", "无法读取图片文件")
                return
                
            # 转换为RGB
            self.original_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.original_image = img_bgr
            self.current_image_path = file_path  # 保存图片路径
            
            # 显示原图
            self.display_image(self.original_image_rgb, self.original_canvas)
            
            # 清空结果图
            self.result_canvas.delete("all")
            self.current_result = None
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """在画布上显示图片（自适应窗口大小）"""
        if image is None:
            return
            
        canvas.delete("all")
        
        # 获取画布尺寸（考虑边框和padding）
        canvas.update()
        canvas_width = max(canvas.winfo_width() - 10, 1)
        canvas_height = max(canvas.winfo_height() - 10, 1)
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400
        
        # 计算缩放比例（允许放大，填满窗口）
        img_height, img_width = image.shape[:2]
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h)  # 允许放大，填满窗口
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 调整图片大小
        if abs(scale - 1.0) > 0.01:  # 如果缩放比例明显不同
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        else:
            resized = image
        
        # 转换为PIL Image
        if len(resized.shape) == 3:
            pil_image = Image.fromarray(resized)
        else:
            pil_image = Image.fromarray(resized).convert("RGB")
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # 居中显示
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # 保持引用
        
        # 绑定窗口大小变化事件
        if not hasattr(canvas, '_resize_bound'):
            def on_canvas_resize(event):
                if hasattr(self, 'original_image_rgb') and self.original_image_rgb is not None:
                    if canvas == self.original_canvas:
                        self.display_image(self.original_image_rgb, canvas)
                    elif canvas == self.result_canvas and self.current_result is not None:
                        self.display_image(self.current_result, canvas)
            canvas.bind('<Configure>', on_canvas_resize)
            canvas._resize_bound = True
    
    def update_algorithm_description(self, algorithm_name: str):
        """更新算法说明"""
        algorithm_descriptions = {
            "ImageCompression": "图像压缩算法，通过JPEG或WebP压缩降低图像质量，模拟真实场景中的压缩伪影。",
            "Downscale": "下采样算法，先将图像缩小再放大回原尺寸，模拟低分辨率图像的细节损失。",
            "GaussNoise": "高斯噪声，添加符合高斯分布的随机噪声，模拟传感器噪声。",
            "Blur": "均匀模糊，使用盒状滤波器对图像进行平滑处理。",
            "GaussianBlur": "高斯模糊，使用高斯滤波器产生自然的模糊效果。",
            "MotionBlur": "运动模糊，模拟相机或物体运动造成的模糊效果。",
            "MedianBlur": "中值模糊，使用中值滤波器，能有效去除椒盐噪声。",
            "ISONoise": "ISO噪声，模拟高ISO设置下的传感器噪声和颜色偏移。",
            "ShotNoise": "泊松噪声（散粒噪声），模拟光子计数的量子噪声。",
            "SaltAndPepper": "椒盐噪声，随机将像素设置为最大值（盐）或最小值（椒）。",
            "Posterize": "色调分离，减少每个颜色通道的位数，产生海报化效果。",
            "Dithering": "抖动算法，通过空间分布模拟更多颜色，减少颜色量化伪影。",
            "RingingOvershoot": "振铃伪影，在边缘附近产生振荡，模拟锐化后的伪影。",
            "MultiplicativeNoise": "乘性噪声，噪声强度与像素值成正比。",
            "AdditiveNoise": "加性噪声，支持多种分布类型（均匀、高斯、拉普拉斯等）。",
            "Defocus": "散焦模糊，模拟镜头失焦效果。",
            "GlassBlur": "玻璃模糊，模拟透过玻璃观察的扭曲效果。",
            "ChromaticAberration": "色差，模拟镜头色散导致的颜色偏移。",
            "CoarseDropout": "粗粒度丢弃，随机丢弃矩形区域，模拟遮挡。",
            "GridDropout": "网格丢弃，按网格模式丢弃区域，产生结构化遮挡。",
            "PixelDropout": "像素丢弃，随机丢弃单个像素。",
            "ChannelDropout": "通道丢弃，随机丢弃整个颜色通道。",
            "ElasticTransform": "弹性变形，产生平滑的弹性形变效果。",
            "GridDistortion": "网格畸变，通过移动网格节点产生畸变。",
            "OpticalDistortion": "光学畸变，模拟镜头畸变（桶形/枕形）。",
            "PiecewiseAffine": "分段仿射，将图像分成网格并分别进行仿射变换。",
            "ThinPlateSpline": "薄板样条，使用薄板样条插值产生平滑变形。"
        }
        
        desc_text = algorithm_descriptions.get(algorithm_name, "该算法的详细说明。")
        self.algorithm_desc_text.config(state=tk.NORMAL)
        self.algorithm_desc_text.delete(1.0, tk.END)
        self.algorithm_desc_text.insert(1.0, desc_text)
        self.algorithm_desc_text.config(state=tk.DISABLED)
    
    def get_algorithm_params(self) -> dict[str, Any]:
        """获取当前算法的参数"""
        algorithm_name = self.algorithm_var.get()
        if algorithm_name not in self.algorithms:
            return {}
            
        algorithm_config = self.algorithms[algorithm_name]
        params = {}
        
        # 特殊处理AdditiveNoise
        if algorithm_name == "AdditiveNoise":
            noise_type = self.param_vars["noise_type"].get()
            spatial_mode = self.param_vars["spatial_mode"].get()
            
            if noise_type == "gaussian":
                noise_params = {
                    "mean_range": (0.0, 0.0),
                    "std_range": (0.05, 0.2)
                }
            elif noise_type == "uniform":
                noise_params = {
                    "ranges": [(-0.1, 0.1)]
                }
            elif noise_type == "laplace":
                noise_params = {
                    "mean_range": (0.0, 0.0),
                    "scale_range": (0.05, 0.2)
                }
            else:
                noise_params = {"ranges": [(-0.1, 0.1)]}
            
            params = {
                "noise_type": noise_type,
                "spatial_mode": spatial_mode,
                "noise_params": noise_params
            }
            return params
        
        for param_name, param_config in algorithm_config["params"].items():
            param_type = param_config["type"]
            var = self.param_vars.get(param_name)
            
            if var is None:
                continue
                
            if param_type in ["range", "range_int"]:
                # 固定值参数：将单个值转换为(min, max)格式（Albumentations需要范围）
                val = var.get()
                
                if param_type == "range_int":
                    val = int(round(val))  # 确保是整数
                    # 对于blur_limit，确保是奇数
                    if "blur" in param_name.lower() and val % 2 == 0:
                        val += 1
                    # 确保在有效范围内
                    val = max(param_config["min"], min(param_config["max"], val))
                    params[param_name] = (val, val)
                else:
                    val = float(val)
                    # 确保在有效范围内
                    val = max(param_config["min"], min(param_config["max"], val))
                    params[param_name] = (val, val)
                        
            elif param_type == "choice":
                params[param_name] = var.get()
                
            elif param_type == "bool":
                params[param_name] = var.get()
        
        return params
    
    def apply_algorithm(self):
        """应用选定的算法"""
        if self.original_image_rgb is None:
            messagebox.showwarning("警告", "请先加载一张图片")
            return
        
        try:
            algorithm_name = self.algorithm_var.get()
            if algorithm_name not in self.algorithms:
                messagebox.showerror("错误", "未知的算法")
                return
            
            algorithm_class = self.algorithms[algorithm_name]["class"]
            params = self.get_algorithm_params()
            
            # 创建变换对象
            transform = algorithm_class(p=1.0, **params)
            
            # 应用变换
            result = transform(image=self.original_image_rgb)
            result_image = result["image"]
            
            # 确保结果是numpy数组
            if not isinstance(result_image, np.ndarray):
                result_image = np.array(result_image)
            
            # 确保是RGB格式
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
            elif result_image.shape[2] == 4:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
            
            self.current_result = result_image
            
            # 显示结果
            self.display_image(result_image, self.result_canvas)
            
        except Exception as e:
            messagebox.showerror("错误", f"应用算法失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def save_result(self):
        """保存结果图片"""
        if self.current_result is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存图片",
            defaultextension=".png",
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 转换为BGR格式保存
            if len(self.current_result.shape) == 3:
                result_bgr = cv2.cvtColor(self.current_result, cv2.COLOR_RGB2BGR)
            else:
                result_bgr = self.current_result
            
            cv2.imwrite(file_path, result_bgr)
            messagebox.showinfo("成功", "图片已保存")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存图片失败: {str(e)}")
    
    def save_random_samples(self):
        """保存范围随机结果"""
        if self.original_image_rgb is None:
            messagebox.showwarning("警告", "请先加载一张图片")
            return
        
        if self.current_image_path is None:
            messagebox.showwarning("警告", "无法确定保存目录，请重新加载图片")
            return
        
        try:
            # 获取样本数量
            sample_count = int(self.sample_count_var.get())
            if sample_count <= 0:
                messagebox.showerror("错误", "样本数量必须大于0")
                return
            if sample_count > 1000:
                if not messagebox.askyesno("确认", f"将生成{sample_count}个样本，可能需要较长时间，是否继续？"):
                    return
        except ValueError:
            messagebox.showerror("错误", "样本数量必须是整数")
            return
        
        algorithm_name = self.algorithm_var.get()
        if algorithm_name not in self.algorithms:
            messagebox.showerror("错误", "未知的算法")
            return
        
        algorithm_config = self.algorithms[algorithm_name]
        
        # 获取参数范围
        range_params = {}
        for param_name, param_config in algorithm_config["params"].items():
            param_type = param_config["type"]
            
            if param_type in ["range", "range_int"]:
                if param_name not in self.range_param_vars:
                    continue
                
                try:
                    var_min = self.range_param_vars[param_name]["min"]
                    var_max = self.range_param_vars[param_name]["max"]
                    
                    min_val = float(var_min.get())
                    max_val = float(var_max.get())
                    
                    # 验证范围
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    
                    # 确保在有效范围内
                    min_val = max(param_config["min"], min(param_config["max"], min_val))
                    max_val = max(param_config["min"], min(param_config["max"], max_val))
                    
                    if param_type == "range_int":
                        min_val = int(round(min_val))
                        max_val = int(round(max_val))
                        # 对于blur_limit，确保是奇数
                        if "blur" in param_name.lower():
                            if min_val % 2 == 0:
                                min_val += 1
                            if max_val % 2 == 0:
                                max_val += 1
                        range_params[param_name] = (min_val, max_val)
                    else:
                        min_val = float(min_val)
                        max_val = float(max_val)
                        range_params[param_name] = (float(min_val), float(max_val))
                        
                except ValueError:
                    messagebox.showerror("错误", f"参数 {param_name} 的范围值无效")
                    return
            elif param_type == "choice":
                # 选择类型使用当前值
                if param_name in self.param_vars:
                    range_params[param_name] = self.param_vars[param_name].get()
            elif param_type == "bool":
                # 布尔类型使用当前值
                if param_name in self.param_vars:
                    range_params[param_name] = self.param_vars[param_name].get()
        
        # 获取保存目录
        import os
        save_dir = os.path.dirname(self.current_image_path)
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        
        # 创建子目录结构：图像名目录/算法英文名子目录
        image_dir = os.path.join(save_dir, base_name)
        output_dir = os.path.join(image_dir, algorithm_name)  # 使用算法英文名作为子目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成并保存样本
        algorithm_class = algorithm_config["class"]
        progress_window = None
        
        try:
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("生成进度")
            progress_window.geometry("400x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="正在生成样本...")
            progress_label.pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_window,
                variable=progress_var,
                maximum=sample_count,
                length=350
            )
            progress_bar.pack(pady=10)
            
            progress_window.update()
            
            # 分离连续参数和离散参数
            continuous_params = {}  # 连续参数（range类型）
            discrete_int_params = {}  # 离散整数参数（range_int类型）
            discrete_fixed_params = {}  # 固定参数（choice, bool类型）
            
            for param_name, param_value in range_params.items():
                if isinstance(param_value, tuple):
                    # 检查参数类型
                    param_config = algorithm_config["params"].get(param_name, {})
                    param_type = param_config.get("type", "range")
                    
                    if param_type == "range":
                        # 连续参数
                        continuous_params[param_name] = param_value
                    else:
                        # 离散参数（range_int）
                        discrete_int_params[param_name] = param_value
                else:
                    # 固定参数（choice, bool）
                    discrete_fixed_params[param_name] = param_value
            
            # 使用拉丁超立方采样生成连续参数的样本
            continuous_param_names = list(continuous_params.keys())
            continuous_samples = None
            
            if len(continuous_param_names) > 0:
                # 生成拉丁超立方采样（LHS）
                # LHS确保每个参数维度都被均匀覆盖
                n_continuous = len(continuous_param_names)
                
                # 生成LHS样本矩阵 (sample_count x n_continuous)
                # 每个参数维度被分成sample_count个区间，每个样本在每个区间中随机采样
                lhs_samples = np.zeros((sample_count, n_continuous))
                
                for i in range(n_continuous):
                    param_name = continuous_param_names[i]
                    min_val, max_val = continuous_params[param_name]
                    
                    # 将参数范围分成sample_count个区间
                    intervals = np.linspace(min_val, max_val, sample_count + 1)
                    
                    # 为每个样本在对应区间中随机采样
                    for j in range(sample_count):
                        lhs_samples[j, i] = np.random.uniform(intervals[j], intervals[j + 1])
                
                # 随机打乱每列的顺序，确保LHS的随机性
                for i in range(n_continuous):
                    np.random.shuffle(lhs_samples[:, i])
                
                continuous_samples = lhs_samples
            
            # 使用类似LHS的策略生成整数参数的样本
            discrete_int_param_names = list(discrete_int_params.keys())
            discrete_int_samples = None
            
            if len(discrete_int_param_names) > 0:
                n_discrete_int = len(discrete_int_param_names)
                discrete_int_samples_matrix = np.zeros((sample_count, n_discrete_int), dtype=int)
                
                for i in range(n_discrete_int):
                    param_name = discrete_int_param_names[i]
                    min_val, max_val = discrete_int_params[param_name]
                    min_val = int(min_val)
                    max_val = int(max_val)
                    
                    # 计算可用的整数值数量
                    num_values = max_val - min_val + 1
                    
                    if num_values <= sample_count:
                        # 如果整数值数量少于样本数，确保每个值至少出现一次
                        # 重复采样直到达到sample_count个
                        values = np.arange(min_val, max_val + 1)
                        # 确保每个值至少出现一次
                        repeated_values = np.tile(values, (sample_count // num_values + 1))[:sample_count]
                        np.random.shuffle(repeated_values)
                        discrete_int_samples_matrix[:, i] = repeated_values
                    else:
                        # 如果整数值数量多于样本数，使用类似LHS的策略
                        # 将范围分成sample_count个区间，每个区间随机选择一个整数值
                        intervals = np.linspace(min_val, max_val, sample_count + 1)
                        for j in range(sample_count):
                            interval_min = int(np.ceil(intervals[j]))
                            interval_max = int(np.floor(intervals[j + 1]))
                            if interval_min <= interval_max:
                                discrete_int_samples_matrix[j, i] = np.random.randint(interval_min, interval_max + 1)
                            else:
                                discrete_int_samples_matrix[j, i] = int(round(intervals[j]))
                        
                        # 对于blur_limit等需要奇数的参数，确保是奇数
                        if "blur" in param_name.lower():
                            for j in range(sample_count):
                                if discrete_int_samples_matrix[j, i] % 2 == 0:
                                    # 如果是偶数，随机选择加1或减1（但要确保在范围内）
                                    if discrete_int_samples_matrix[j, i] + 1 <= max_val:
                                        discrete_int_samples_matrix[j, i] += 1
                                    elif discrete_int_samples_matrix[j, i] - 1 >= min_val:
                                        discrete_int_samples_matrix[j, i] -= 1
                
                # 随机打乱每列的顺序
                for i in range(n_discrete_int):
                    np.random.shuffle(discrete_int_samples_matrix[:, i])
                
                discrete_int_samples = discrete_int_samples_matrix
            
            # 用于去重的集合（存储参数组合的哈希值）
            seen_param_combinations = set()
            saved_count = 0
            max_attempts = sample_count * 10  # 最大尝试次数，避免无限循环
            attempt_count = 0
            
            while saved_count < sample_count and attempt_count < max_attempts:
                attempt_count += 1
                
                # 生成随机参数
                random_params = {}
                param_values_for_filename = {}  # 用于文件名的参数值
                
                # 处理连续参数（使用LHS采样）
                if continuous_samples is not None:
                    sample_idx = saved_count % sample_count  # 循环使用LHS样本
                    for i, param_name in enumerate(continuous_param_names):
                        random_val = float(continuous_samples[sample_idx, i])
                        min_val, max_val = continuous_params[param_name]
                        # 确保在范围内
                        random_val = max(min_val, min(max_val, random_val))
                        random_params[param_name] = (random_val, random_val)
                        param_values_for_filename[param_name] = random_val
                
                # 处理离散整数参数（使用改进的采样策略）
                if discrete_int_samples is not None:
                    sample_idx = saved_count % sample_count  # 循环使用采样样本
                    for i, param_name in enumerate(discrete_int_param_names):
                        random_val = int(discrete_int_samples[sample_idx, i])
                        random_params[param_name] = (random_val, random_val)
                        param_values_for_filename[param_name] = random_val
                
                # 处理固定参数（choice, bool）
                for param_name, param_value in discrete_fixed_params.items():
                    random_params[param_name] = param_value
                    param_values_for_filename[param_name] = param_value
                
                # 生成参数组合的哈希值用于去重
                # 将参数值转换为可哈希的元组，对浮点数进行适当舍入以避免精度问题
                def normalize_value(v):
                    """标准化参数值用于去重"""
                    if isinstance(v, tuple):
                        v = v[0]
                    if isinstance(v, (float, np.floating)):
                        # 浮点数保留4位小数，避免精度问题导致的重复
                        return round(float(v), 4)
                    elif isinstance(v, (int, np.integer)):
                        return int(v)
                    elif isinstance(v, bool):
                        return bool(v)
                    else:
                        return str(v)
                
                param_tuple = tuple(sorted((k, normalize_value(v)) 
                                          for k, v in param_values_for_filename.items()))
                param_hash = hash(param_tuple)
                
                # 检查是否重复
                if param_hash in seen_param_combinations:
                    continue  # 跳过重复的参数组合
                
                # 记录这个参数组合
                seen_param_combinations.add(param_hash)
                
                # 创建变换对象
                transform = algorithm_class(p=1.0, **random_params)
                
                # 应用变换
                result = transform(image=self.original_image_rgb)
                result_image = result["image"]
                
                # 确保结果是numpy数组
                if not isinstance(result_image, np.ndarray):
                    result_image = np.array(result_image)
                
                # 确保是RGB格式
                if len(result_image.shape) == 2:
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
                elif result_image.shape[2] == 4:
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGBA2RGB)
                
                # 构建文件名：原始图像名_参数a名称(值)_参数b名称(值)...（使用纯英文）
                filename_parts = [base_name]
                for param_name, param_val in sorted(param_values_for_filename.items()):
                    # 使用英文参数名称（直接使用参数名，或从配置中获取英文标签）
                    param_config = algorithm_config["params"].get(param_name, {})
                    # 使用参数名本身作为英文名称（参数名已经是英文）
                    param_label = param_name
                    
                    # 格式化参数值
                    if isinstance(param_val, tuple):
                        # 如果是元组，取第一个值（因为我们已经转换为(value, value)格式）
                        param_val = param_val[0]
                    
                    if isinstance(param_val, (int, np.integer)):
                        param_str = f"{param_label}({int(param_val)})"
                    elif isinstance(param_val, (float, np.floating)):
                        # 浮点数保留3位小数，去除末尾的0
                        param_str = f"{param_label}({param_val:.3f})".rstrip('0').rstrip('.')
                    elif isinstance(param_val, bool):
                        param_str = f"{param_label}({str(param_val)})"
                    else:
                        param_str = f"{param_label}({str(param_val)})"
                    
                    # 清理文件名中的非法字符
                    param_str = param_str.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")
                    filename_parts.append(param_str)
                
                filename_base = "_".join(filename_parts)
                
                # 如果文件名太长，截断（Windows限制255字符，保留扩展名）
                max_filename_length = 200  # 留一些余量
                extension = ".png"
                if len(filename_base) > max_filename_length - len(extension):
                    # 保留基础名称和扩展名，截断中间部分
                    base_len = len(base_name) + len(extension)
                    available_len = max_filename_length - base_len - 10  # 留10个字符的余量
                    if available_len > 0:
                        # 截断参数部分
                        param_part = "_".join(filename_parts[1:])
                        if len(param_part) > available_len:
                            param_part = param_part[:available_len]
                        filename_base = f"{base_name}_{param_part}"
                    else:
                        # 如果基础名称本身就很长，只保留基础名称
                        filename_base = base_name[:max_filename_length - len(extension)]
                
                # 检查文件名是否已存在，如果存在则添加序号
                filename = filename_base + extension
                filepath = os.path.join(output_dir, filename)
                counter = 1
                while os.path.exists(filepath):
                    filename = f"{filename_base}_{counter:03d}{extension}"
                    filepath = os.path.join(output_dir, filename)
                    counter += 1
                
                # 转换为BGR格式保存
                if len(result_image.shape) == 3:
                    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                else:
                    result_bgr = result_image
                
                cv2.imwrite(filepath, result_bgr)
                saved_count += 1
                
                # 更新进度
                progress_var.set(saved_count)
                progress_label.config(text=f"正在生成样本... ({saved_count}/{sample_count})")
                progress_window.update()
            
            # 检查是否成功生成足够的样本
            if saved_count < sample_count:
                messagebox.showwarning(
                    "警告",
                    f"由于参数空间限制，只生成了 {saved_count} 个唯一样本（请求 {sample_count} 个）。\n"
                    f"这可能是因为参数范围太小或离散参数组合有限。"
                )
            
            progress_window.destroy()
            messagebox.showinfo(
                "成功", 
                f"已成功生成并保存 {saved_count} 个样本到:\n{output_dir}"
            )
            
        except Exception as e:
            if progress_window:
                progress_window.destroy()
            messagebox.showerror("错误", f"生成样本失败: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    root = tk.Tk()
    app = DegradationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

