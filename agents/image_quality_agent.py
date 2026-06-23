"""
问题1：无参考图像质量评价模型
"""

import numpy as np
from PIL import Image
import clip
import torch
import cv2

class ImageQualityAgent:
    def __init__(self):
        """初始化图像质量评估Agent"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ 使用设备: {self.device}")
        
        # 加载CLIP模型用于语义相似度计算
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("✓ CLIP模型加载成功")
    
    def semantic_fidelity_metric(self, prompt: str, image_path: str):
        """
        计算语义保真度指标 - 基于CLIP语义相似度
        """
        try:
            # 加载并预处理图像
            image = Image.open(image_path)
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # 文本编码
            text = clip.tokenize([prompt]).to(self.device)
            
            # 计算CLIP相似度
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text)
                
                # 归一化特征
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # 计算余弦相似度
                similarity = (image_features @ text_features.T).item()
            
            return {
                "semantic_similarity": similarity,
                "metric_explanation": "CLIP语义相似度，范围[-1, 1]，值越大表示语义保真度越高"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def technical_quality_metric(self, image_path: str):
        """
        计算技术质量指标 - 基于清晰度、噪声和频域分析
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "无法读取图像"}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. 清晰度指标（拉普拉斯方差）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. 噪声估计
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.std(gray - blurred)
            
            # 3. 频域分析（FFT高频能量）
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_energy = np.mean(magnitude_spectrum[magnitude_spectrum > np.percentile(magnitude_spectrum, 75)])
            
            return {
                "clarity": float(laplacian_var),
                "noise_level": float(noise),
                "high_freq_energy": float(high_freq_energy),
                "metric_explanation": "清晰度、噪声水平、高频能量"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def structural_integrity_metric(self, image_path: str):
        """
        计算结构完整性指标 - 基于边缘连续性和形状规则性
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "无法读取图像"}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Canny边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 2. 边缘密度
            edge_density = np.sum(edges > 0) / edges.size
            
            # 3. 形状规则性
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour_areas = [cv2.contourArea(c) for c in contours]
                shape_regularity = np.std(contour_areas) / (np.mean(contour_areas) + 1e-6)
            else:
                shape_regularity = 0.0
            
            return {
                "edge_density": float(edge_density),
                "shape_regularity": float(shape_regularity),
                "metric_explanation": "边缘密度、形状规则性"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def weighted_quality_index(self, semantic_score: float, technical_score: float, structural_score: float):
        """
        计算加权综合质量指数
        """
        # 定义权重
        weights = {
            "semantic": 0.4,
            "technical": 0.35,
            "structural": 0.25
        }
        
        # 归一化各指标到[0,1]范围
        semantic_norm = (semantic_score + 1) / 2
        technical_norm = min(technical_score / 1000, 1)
        structural_norm = min(structural_score * 10, 1)
        
        # 计算综合指数
        quality_index = (
            weights["semantic"] * semantic_norm +
            weights["technical"] * technical_norm +
            weights["structural"] * structural_norm
        )
        
        return {
            "quality_index": float(quality_index),
            "weights": weights,
            "normalized_scores": {
                "semantic": float(semantic_norm),
                "technical": float(technical_norm),
                "structural": float(structural_norm)
            },
            "interpretation": "质量指数范围[0,1]，值越大表示图像质量越好"
        }