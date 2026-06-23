"""
DeepSeek Agent — 纯 requests + dotenv 实现（零外部依赖）
"""

import os
import requests
import json
from dotenv import load_dotenv

# 加载环境变量（自动查找项目根目录下的 .env）
load_dotenv()

class DeepSeekAgent:
    """
    使用 DeepSeek REST API 的轻量级 Agent
    支持 chat 模式，兼容 deepseek-chat / deepseek-coder
    """
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
        if not self.api_key:
            raise RuntimeError(
                "❌ 缺少 DeepSeek API Key！请在 .env 文件中设置 DEEPSEEK_API_KEY"
            )
        
        # 验证基础配置
        print(f"✓ DeepSeekAgent 初始化成功 | 模型: {self.model}")

    def chat(self, user_message: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2048):
        """
        发送单轮对话请求到 DeepSeek API
        
        Args:
            user_message (str): 用户输入内容
            system_prompt (str, optional): 系统角色提示词
            temperature (float): 采样温度（0~2），默认 0.7
            max_tokens (int): 最大生成 token 数，默认 2048
        
        Returns:
            str: 模型返回的文本，或错误信息
        """
        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False  # 同步模式
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30  # 超时保护
            )
            response.raise_for_status()  # 抛出 HTTP 错误

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content.strip()

        except requests.exceptions.RequestException as e:
            return f"⚠️ 网络请求失败: {type(e).__name__} - {e}"
        except KeyError as e:
            return f"⚠️ 响应格式异常: 缺少字段 {e}，原始响应: {response.text[:200]}"
        except Exception as e:
            return f"⚠️ 未知错误: {type(e).__name__}: {e}"

    def __repr__(self):
        return f"<DeepSeekAgent(model='{self.model}')>"