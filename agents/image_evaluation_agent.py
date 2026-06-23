"""
问题2：图像质量评估算法
"""

import pandas as pd
import os
import numpy as np
from .image_quality_agent import ImageQualityAgent
from .deepseek_tools import DeepSeekAgent

class ImageEvaluationAgent:
    def __init__(self):
        """初始化图像评估Agent"""
        self.quality_agent = ImageQualityAgent()
        self.deepseek_agent = DeepSeekAgent()
    
    def evaluate_single_image(self, image_path: str, prompt: str = ""):
        """
        评估单张图像的质量
        """
        results = {
            "image_name": os.path.basename(image_path),
            "image_path": image_path
        }
        
        # 1. 语义保真度评估（如果有提示词）
        if prompt:
            semantic_result = self.quality_agent.semantic_fidelity_metric(prompt, image_path)
            if "error" not in semantic_result:
                results["semantic_similarity"] = semantic_result["semantic_similarity"]
            else:
                results["semantic_similarity"] = None
        else:
            results["semantic_similarity"] = None
        
        # 2. 技术质量评估
        technical_result = self.quality_agent.technical_quality_metric(image_path)
        if "error" not in technical_result:
            results["clarity"] = technical_result["clarity"]
            results["noise_level"] = technical_result["noise_level"]
            results["high_freq_energy"] = technical_result["high_freq_energy"]
        else:
            results["clarity"] = None
            results["noise_level"] = None
            results["high_freq_energy"] = None
        
        # 3. 结构完整性评估
        structural_result = self.quality_agent.structural_integrity_metric(image_path)
        if "error" not in structural_result:
            results["edge_density"] = structural_result["edge_density"]
            results["shape_regularity"] = structural_result["shape_regularity"]
        else:
            results["edge_density"] = None
            results["shape_regularity"] = None
        
        # 4. 计算综合质量指数
        if results["semantic_similarity"] is not None:
            quality_index_result = self.quality_agent.weighted_quality_index(
                results["semantic_similarity"],
                results["clarity"],
                results["edge_density"]
            )
            results["quality_index"] = quality_index_result["quality_index"]
        else:
            # 如果没有提示词，使用技术+结构指标
            technical_norm = min(results["clarity"] / 1000, 1) if results["clarity"] else 0
            structural_norm = min(results["edge_density"] * 10, 1) if results["edge_density"] else 0
            results["quality_index"] = (technical_norm + structural_norm) / 2
        
        return results
    
    def evaluate_batch_images(self, image_folder: str, prompts_dict: dict = None):
        """
        批量评估图像质量
        """
        if prompts_dict is None:
            prompts_dict = {}
        
        results = []
        
        # 遍历图像文件夹
        for image_file in os.listdir(image_folder):
            if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(image_folder, image_file)
                image_name = os.path.splitext(image_file)[0]
                
                # 获取对应提示词
                prompt = prompts_dict.get(image_name, "")
                
                # 评估图像
                result = self.evaluate_single_image(image_path, prompt)
                results.append(result)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def classify_content_type(self, image_name: str):
        """
        分类内容类型
        """
        image_name_lower = image_name.lower()
        
        if any(word in image_name_lower for word in ['landscape', '风景', '自然', '山水']):
            return "写实风景"
        elif any(word in image_name_lower for word in ['portrait', '人物', '人像', '肖像']):
            return "人物肖像"
        elif any(word in image_name_lower for word in ['illustration', '插画', '艺术', '绘画']):
            return "艺术插画"
        elif any(word in image_name_lower for word in ['product', '产品', '商品', '渲染']):
            return "产品渲染"
        else:
            return "未知"
    
    def analyze_sensitivity(self, content_type: str):
        """
        分析不同内容类型对各维度指标的敏感性
        """
        sensitivity_weights = {}
        
        if content_type == "写实风景":
            sensitivity_weights = {
                "semantic_sensitivity": 0.3,
                "technical_sensitivity": 0.5,
                "structural_sensitivity": 0.2
            }
        elif content_type == "人物肖像":
            sensitivity_weights = {
                "semantic_sensitivity": 0.4,
                "technical_sensitivity": 0.3,
                "structural_sensitivity": 0.3
            }
        elif content_type == "艺术插画":
            sensitivity_weights = {
                "semantic_sensitivity": 0.6,
                "technical_sensitivity": 0.2,
                "structural_sensitivity": 0.2
            }
        elif content_type == "产品渲染":
            sensitivity_weights = {
                "semantic_sensitivity": 0.3,
                "technical_sensitivity": 0.6,
                "structural_sensitivity": 0.1
            }
        else:
            sensitivity_weights = {
                "semantic_sensitivity": 0.4,
                "technical_sensitivity": 0.35,
                "structural_sensitivity": 0.25
            }
        
        return sensitivity_weights
    
    def deepseek_quality_analysis(self, image_metrics: dict, content_type: str):
        """
        使用DeepSeek进行质量分析
        """
        analysis = self.deepseek_agent.analyze_image_quality(image_metrics, content_type)
        return analysis
    
    def cross_model_comparison(self, model_results: dict):
        """
        跨模型对比分析评估结果的可靠性
        """
        comparison = {}
        
        for model_name, results_df in model_results.items():
            if 'quality_index' in results_df.columns:
                quality_scores = results_df['quality_index'].dropna()
                
                if len(quality_scores) > 0:
                    comparison[model_name] = {
                        "mean_quality": float(np.mean(quality_scores)),
                        "std_quality": float(np.std(quality_scores)),
                        "max_quality": float(np.max(quality_scores)),
                        "min_quality": float(np.min(quality_scores)),
                        "sample_count": len(quality_scores),
                        "reliability_score": float(1 / (np.std(quality_scores) + 1e-6))
                    }
        
        return comparison