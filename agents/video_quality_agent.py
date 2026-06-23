"""
问题3：视频时序质量评估
"""

import cv2
import numpy as np
from .deepseek_tools import DeepSeekAgent

class VideoQualityAgent:
    def __init__(self):
        """初始化视频质量评估Agent"""
        self.deepseek_agent = DeepSeekAgent()
    
    def optical_flow_continuity(self, video_path: str):
        """
        计算光流连续性指标
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            prev_frame = None
            flow_magnitudes = []
            flow_angles = []
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # 计算光流
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # 计算光流幅度和角度
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    flow_magnitudes.append(np.mean(magnitude))
                    flow_angles.append(np.mean(angle))
                
                prev_frame = gray
                frame_count += 1
            
            cap.release()
            
            # 计算光流的平滑程度
            if len(flow_magnitudes) > 1:
                magnitude_std = np.std(flow_magnitudes)
                angle_std = np.std(flow_angles)
                smoothness = 1 / (magnitude_std + angle_std + 1e-6)
            else:
                smoothness = 0.0
            
            return {
                "flow_smoothness": float(smoothness),
                "magnitude_std": float(magnitude_std) if len(flow_magnitudes) > 1 else 0.0,
                "angle_std": float(angle_std) if len(flow_angles) > 1 else 0.0,
                "frame_count": frame_count,
                "metric_explanation": "光流平滑度，值越大表示运动越连续"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def content_consistency(self, video_path: str):
        """
        计算内容一致性指标
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            orb = cv2.ORB_create()
            prev_keypoints = None
            prev_descriptors = None
            
            match_scores = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                
                if prev_descriptors is not None and descriptors is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(prev_descriptors, descriptors)
                    
                    if len(matches) > 0:
                        match_score = len(matches) / max(len(prev_keypoints), len(keypoints))
                        match_scores.append(match_score)
                
                prev_keypoints = keypoints
                prev_descriptors = descriptors
                frame_count += 1
            
            cap.release()
            
            # 计算一致性
            if len(match_scores) > 0:
                mean_consistency = np.mean(match_scores)
                consistency_stability = 1 / (np.std(match_scores) + 1e-6)
                overall_consistency = mean_consistency * consistency_stability
            else:
                overall_consistency = 0.0
            
            # ⭐ 归一化处理：限制在 [0, 1] 范围
            overall_consistency = min(overall_consistency, 1.0)
            
            return {
                "content_consistency": float(overall_consistency),
                "mean_match_score": float(np.mean(match_scores)) if match_scores else 0.0,
                "consistency_stability": float(consistency_stability) if match_scores else 0.0,
                "frame_count": frame_count,
                "metric_explanation": "内容一致性，值越大表示物体特征越稳定"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def flicker_detection(self, video_path: str):
        """
        闪烁检测
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            brightness_values = []
            color_variations = []
            
            prev_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 计算亮度
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                brightness = np.mean(yuv[:, :, 0])
                brightness_values.append(brightness)
                
                # 计算色彩变化
                if prev_frame is not None:
                    color_diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                    color_variations.append(color_diff)
                
                prev_frame = frame
            
            cap.release()
            
            # 检测突变
            if len(brightness_values) > 1:
                brightness_diff = np.diff(brightness_values)
                brightness_flicker = np.sum(np.abs(brightness_diff) > np.std(brightness_values) * 2)
                color_flicker = np.sum(np.array(color_variations) > np.std(color_variations) * 2) if color_variations else 0
                total_frames = len(brightness_values)
                flicker_degree = (brightness_flicker + color_flicker) / total_frames
            else:
                flicker_degree = 0.0
            
            return {
                "flicker_degree": float(flicker_degree),
                "brightness_flicker_count": int(brightness_flicker) if len(brightness_values) > 1 else 0,
                "color_flicker_count": int(color_flicker),
                "metric_explanation": "闪烁程度，值越小表示闪烁越少"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def temporal_instability_condition(self, flow_smoothness: float, content_consistency: float, flicker_degree: float):
        """
        推导产生时序失稳的必要条件
        """
        instability_conditions = {
            "flow_condition": flow_smoothness < 0.5,
            "consistency_condition": content_consistency < 0.3,
            "flicker_condition": flicker_degree > 0.2,
        }
        
        is_unstable = (
            instability_conditions["flow_condition"] or
            instability_conditions["consistency_condition"] or
            instability_conditions["flicker_condition"]
        )
        
        return {
            "is_temporally_unstable": is_unstable,
            "conditions": instability_conditions,
            "explanation": "时序失稳的必要条件：光流平滑度<0.5 或 内容一致性<0.3 或 闪烁程度>0.2"
        }
    
    def video_quality_model(self, video_path: str):
        """
        视频质量综合评估模型
        """
        try:
            # 1. 计算光流连续性
            flow_result = self.optical_flow_continuity(video_path)
            if "error" in flow_result:
                return flow_result
            
            # 2. 计算内容一致性
            consistency_result = self.content_consistency(video_path)
            if "error" in consistency_result:
                return consistency_result
            
            # 3. 计算闪烁程度
            flicker_result = self.flicker_detection(video_path)
            if "error" in flicker_result:
                return flicker_result
            
            # 4. 判断时序失稳
            instability_result = self.temporal_instability_condition(
                flow_result["flow_smoothness"],
                consistency_result["content_consistency"],
                flicker_result["flicker_degree"]
            )
            
            # ⭐ 归一化处理：各指标限制在 [0, 1] 范围
            optical_flow_quality = min(flow_result["flow_smoothness"], 1.0)
            content_consistency_quality = min(consistency_result["content_consistency"], 1.0)
            flicker_quality = 1 - flicker_result["flicker_degree"]
            
            # 5. 计算综合视频质量指数
            temporal_quality = (
                optical_flow_quality * 0.4 +
                content_consistency_quality * 0.4 +
                flicker_quality * 0.2
            )
            
            # 如果时序失稳，质量指数大幅降低
            if instability_result["is_temporally_unstable"]:
                temporal_quality *= 0.5
            
            return {
                "video_path": video_path,
                "optical_flow_quality": optical_flow_quality,
                "content_consistency_quality": content_consistency_quality,
                "flicker_quality": flicker_quality,
                "temporal_quality": temporal_quality,
                "is_temporally_unstable": instability_result["is_temporally_unstable"],
                "instability_conditions": instability_result["conditions"],
                "overall_video_quality": temporal_quality,
                "frame_count": flow_result["frame_count"]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def deepseek_video_analysis(self, video_metrics: dict):
        """
        使用DeepSeek进行视频质量分析
        """
        analysis = self.deepseek_agent.analyze_video_quality(video_metrics)
        return analysis