"""
AIGC质量评估与参数优化系统 - 主程序
"""

import os
import sys
from agents.image_quality_agent import ImageQualityAgent
from agents.image_evaluation_agent import ImageEvaluationAgent
from agents.video_quality_agent import VideoQualityAgent
import pandas as pd
import json

def main():
    """主程序入口"""
    print("=" * 60)
    print("AIGC质量评估与参数优化系统 (DeepSeek版本)")
    print("=" * 60)
    print()
    
    # 确保结果文件夹存在
    os.makedirs("results", exist_ok=True)
    
    # ==================== 问题1：图像质量评价模型 ====================
    print("【问题1】建立无参考图像质量评价模型...")
    print("-" * 60)
    
    image_quality_agent = ImageQualityAgent()
    
    # 示例：评估单张测试图像
    sample_image = "data/sample_test.jpg"
    sample_prompt = "一只可爱的橘色猫咪在阳光下打盹"
    
    if os.path.exists(sample_image):
        print(f"\n评估示例图像: {sample_image}")
        print(f"提示词: {sample_prompt}")
        
        # 语义保真度
        semantic_result = image_quality_agent.semantic_fidelity_metric(sample_prompt, sample_image)
        print(f"\n1. 语义保真度:")
        for k, v in semantic_result.items():
            print(f"   {k}: {v}")
        
        # 技术质量
        technical_result = image_quality_agent.technical_quality_metric(sample_image)
        print(f"\n2. 技术质量:")
        for k, v in technical_result.items():
            print(f"   {k}: {v}")
        
        # 结构完整性
        structural_result = image_quality_agent.structural_integrity_metric(sample_image)
        print(f"\n3. 结构完整性:")
        for k, v in structural_result.items():
            print(f"   {k}: {v}")
        
        # 综合质量指数
        if "semantic_similarity" in semantic_result:
            quality_index = image_quality_agent.weighted_quality_index(
                semantic_result["semantic_similarity"],
                technical_result["clarity"],
                structural_result["edge_density"]
            )
            print(f"\n4. 综合质量指数:")
            for k, v in quality_index.items():
                print(f"   {k}: {v}")
    else:
        print(f"\n⚠️  提示: 请将测试图像放入 data/sample_test.jpg")
    
    print()
    
    # ==================== 问题2：图像质量评估算法 ====================
    print("【问题2】图像质量评估算法...")
    print("-" * 60)
    
    image_eval_agent = ImageEvaluationAgent()
    
    # 准备测试数据
    image_folder = "data/attachment1"
    
    if os.path.exists(image_folder):
        print(f"\n批量评估图像文件夹: {image_folder}")
        
        # 创建示例提示词字典
        prompts_dict = {
            "image1": "写实风景图像",
            "image2": "人物肖像照片", 
            "image3": "艺术插画作品",
            "image4": "产品渲染图",
        }
        
        # 批量评估
        print("\n正在评估图像...")
        evaluation_results = image_eval_agent.evaluate_batch_images(image_folder, prompts_dict)
        
        print(f"\n✓ 评估完成! 共评估 {len(evaluation_results)} 张图像")
        print("\n评估结果预览:")
        print("-" * 60)
        print(evaluation_results.to_string())
        
        # 保存结果
        output_file = "results/image_evaluation_results.csv"
        evaluation_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 评估结果已保存至: {output_file}")
        
        # 内容类型分类
        print("\n内容类型分类:")
        print("-" * 60)
        evaluation_results['content_type'] = evaluation_results['image_name'].apply(
            image_eval_agent.classify_content_type
        )
        print(evaluation_results[['image_name', 'content_type']].to_string())
        
        # 敏感性分析
        print("\n敏感性分析示例:")
        print("-" * 60)
        for content_type in evaluation_results['content_type'].unique():
            if content_type != "未知":
                sensitivity = image_eval_agent.analyze_sensitivity(content_type)
                print(f"{content_type}:")
                for k, v in sensitivity.items():
                    print(f"  {k}: {v}")
        
    else:
        print(f"\n⚠️  提示: 请将附件1的图像放入 data/attachment1/ 文件夹")
    
    print()
    
    # ==================== 问题3：视频时序质量评估 ====================
    print("【问题3】视频时序质量评估...")
    print("-" * 60)
    
    video_quality_agent = VideoQualityAgent()
    
    video_path = "data/attachment2/car_traffic.mp4"
    
    if os.path.exists(video_path):
        print(f"\n评估视频: {video_path}")
        print("\n正在分析视频质量...")
        
        # 视频质量评估
        video_quality_result = video_quality_agent.video_quality_model(video_path)
        
        print("\n✓ 视频质量评估结果:")
        print("-" * 60)
        for key, value in video_quality_result.items():
            print(f"  {key}: {value}")
        
        # 保存结果
        output_file = "results/video_quality_result.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(video_quality_result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 视频评估结果已保存至: {output_file}")
        
    else:
        print(f"\n⚠️  提示: 请将附件2的视频放入 data/attachment2/car_traffic.mp4")
    
    print()
    print("=" * 60)
    print("✓ 评估完成!")
    print("=" * 60)
    print("\n结果文件:")
    print("  - 图像评估: results/image_evaluation_results.csv")
    print("  - 视频评估: results/video_quality_result.json")

if __name__ == "__main__":
    main()