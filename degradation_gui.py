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
            width=30
        )
        self.algorithm_combo.pack(fill=tk.X, pady=(0, 10))
        self.algorithm_combo.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        # 参数控制区域
        self.param_frame = ttk.Frame(control_panel)
        self.param_frame.pack(fill=tk.BOTH, expand=True)
        
        # 应用按钮
        ttk.Button(
            control_panel, 
            text="应用算法", 
            command=self.apply_algorithm
        ).pack(fill=tk.X, pady=(10, 0))
        
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
        self.algorithms = {
            "ImageCompression": {
                "class": A.ImageCompression,
                "params": {
                    "quality_range": {"type": "range", "min": 1, "max": 100, "default": (50, 80), "label": "质量范围"},
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
                    "std_range": {"type": "range", "min": 0.0, "max": 50.0, "default": (5.0, 25.0), "label": "标准差范围"},
                    "mean_range": {"type": "range", "min": -10.0, "max": 10.0, "default": (0.0, 0.0), "label": "均值范围"},
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
            }
        }
        
        # 更新算法下拉列表
        self.algorithm_combo["values"] = list(self.algorithms.keys())
        self.on_algorithm_change()
        
    def on_algorithm_change(self, event=None):
        """当算法改变时更新参数控制"""
        # 清除旧的参数控件
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        
        algorithm_name = self.algorithm_var.get()
        if algorithm_name not in self.algorithms:
            return
            
        algorithm_config = self.algorithms[algorithm_name]
        self.param_vars = {}
        
        # 创建参数控件
        for param_name, param_config in algorithm_config["params"].items():
            param_type = param_config["type"]
            param_label = param_config.get("label", param_name)
            param_default = param_config.get("default")
            
            # 标签
            label_frame = ttk.Frame(self.param_frame)
            label_frame.pack(fill=tk.X, pady=5)
            ttk.Label(label_frame, text=param_label + ":").pack(side=tk.LEFT)
            
            if param_type == "range":
                # 范围滑块（两个值）
                var_min = tk.DoubleVar(value=param_default[0] if isinstance(param_default, tuple) else param_default)
                var_max = tk.DoubleVar(value=param_default[1] if isinstance(param_default, tuple) else param_default)
                
                min_frame = ttk.Frame(self.param_frame)
                min_frame.pack(fill=tk.X, pady=2)
                ttk.Label(min_frame, text="  最小值:", width=10).pack(side=tk.LEFT)
                scale_min = ttk.Scale(
                    min_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_min,
                    orient=tk.HORIZONTAL
                )
                scale_min.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_min = ttk.Label(min_frame, textvariable=var_min, width=8)
                label_min.pack(side=tk.LEFT)
                
                max_frame = ttk.Frame(self.param_frame)
                max_frame.pack(fill=tk.X, pady=2)
                ttk.Label(max_frame, text="  最大值:", width=10).pack(side=tk.LEFT)
                scale_max = ttk.Scale(
                    max_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_max,
                    orient=tk.HORIZONTAL
                )
                scale_max.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_max = ttk.Label(max_frame, textvariable=var_max, width=8)
                label_max.pack(side=tk.LEFT)
                
                self.param_vars[param_name] = (var_min, var_max)
                
            elif param_type == "range_int":
                # 整数范围滑块
                var_min = tk.IntVar(value=param_default[0] if isinstance(param_default, tuple) else param_default)
                var_max = tk.IntVar(value=param_default[1] if isinstance(param_default, tuple) else param_default)
                
                min_frame = ttk.Frame(self.param_frame)
                min_frame.pack(fill=tk.X, pady=2)
                ttk.Label(min_frame, text="  最小值:", width=10).pack(side=tk.LEFT)
                scale_min = ttk.Scale(
                    min_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_min,
                    orient=tk.HORIZONTAL
                )
                scale_min.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_min = ttk.Label(min_frame, textvariable=var_min, width=8)
                label_min.pack(side=tk.LEFT)
                
                max_frame = ttk.Frame(self.param_frame)
                max_frame.pack(fill=tk.X, pady=2)
                ttk.Label(max_frame, text="  最大值:", width=10).pack(side=tk.LEFT)
                scale_max = ttk.Scale(
                    max_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    variable=var_max,
                    orient=tk.HORIZONTAL
                )
                scale_max.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                label_max = ttk.Label(max_frame, textvariable=var_max, width=8)
                label_max.pack(side=tk.LEFT)
                
                self.param_vars[param_name] = (var_min, var_max)
                
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
            
            # 显示原图
            self.display_image(self.original_image_rgb, self.original_canvas)
            
            # 清空结果图
            self.result_canvas.delete("all")
            self.current_result = None
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def display_image(self, image: np.ndarray, canvas: tk.Canvas):
        """在画布上显示图片"""
        if image is None:
            return
            
        canvas.delete("all")
        
        # 获取画布尺寸
        canvas.update()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400
        
        # 计算缩放比例
        img_height, img_width = image.shape[:2]
        scale_w = canvas_width / img_width
        scale_h = canvas_height / img_height
        scale = min(scale_w, scale_h, 1.0)  # 不放大，只缩小
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 调整图片大小
        if scale < 1.0:
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
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
                # 范围参数
                var_min, var_max = var
                min_val = var_min.get()
                max_val = var_max.get()
                
                # 确保min <= max
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                
                # 对于整数范围，确保是奇数（对于blur_limit）
                if param_type == "range_int" and "blur" in param_name.lower():
                    min_val = int(min_val)
                    max_val = int(max_val)
                    if min_val % 2 == 0:
                        min_val += 1
                    if max_val % 2 == 0:
                        max_val += 1
                    params[param_name] = (min_val, max_val)
                else:
                    if param_type == "range_int":
                        params[param_name] = (int(min_val), int(max_val))
                    else:
                        params[param_name] = (float(min_val), float(max_val))
                        
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


def main():
    """主函数"""
    root = tk.Tk()
    app = DegradationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

