# AIGC_Agent - AI生成内容质量评估系统

## 项目地址：https://github.com/Hellowwwwgiy/AIGC_Agent

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于人工智能的图像和视频质量评估代理系统，用于自动化评估AIGC（AI生成内容）的质量指标。

## 🌟 项目简介

AIGC_Agent 是一个专门用于评估AI生成内容质量的智能代理系统。它能够：
- 📸 **图像质量评估**：分析图像的清晰度、色彩、构图等技术指标
- 🎥 **视频质量评估**：评估视频的流畅度、稳定性、画质等维度
- 🤖 **自动化评分**：为生成内容提供客观的质量评分

## ✨ 核心功能

### 1. 图像质量评估代理 (`ImageQualityAgent`)
- 评估图像的技术质量指标
- 分析图像的清晰度、对比度、色彩平衡
- 提供综合质量评分

### 2. 视频质量评估代理 (`VideoQualityAgent`)
- 评估视频的流畅度和稳定性
- 分析视频画质和压缩质量
- 提供视频质量综合评分

### 3. 模块化设计
- 独立的代理模块，易于扩展
- 统一的接口设计，方便集成
- 支持自定义评估指标

## 📂 项目结构
```
AIGC_Agent/
├── main.py # 主程序入口
├── agents/ # 评估代理模块
│ ├── init.py
│ ├── image_quality_agent.py # 图像质量评估代理
│ ├──deepseek_tools.py # 工具函数
│ ├──image_evaluation_agent.py — 图像内容综合评估
| └──video_quality_agent.py # 视频质量评估代理
├── requirements.txt # 依赖包列表
├── .gitignore # Git忽略文件配置
└── README.md # 项目说明文档
```
## 🔧 虚拟环境配置

### 创建虚拟环境
```
bash
python -m venv venv
```
### 激活虚拟环境
1. Windows:
.\venv\Scripts\activate
2. Linux/Mac:
source venv/bin/activate
激活后命令行会显示 (venv) 前缀。

### 安装依赖
```bash
pip install opencv-python numpy
```

### 运行项目
```python 
main.py
```

## 联系方式
GitHub: @Hellowwwwgiy
