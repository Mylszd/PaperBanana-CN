"""
Provider 抽象基类
定义所有 API 提供商必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseProvider(ABC):
    """所有 API Provider 的抽象基类"""

    @abstractmethod
    async def generate_text(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = 50000,
        max_attempts: int = 5,
        retry_delay: float = 5,
        error_context: str = "",
    ) -> List[str]:
        """
        文本生成接口

        Args:
            model_name: 模型名称
            contents: 通用内容列表（文本和图片混合）
            system_prompt: 系统提示词
            temperature: 温度参数
            max_output_tokens: 最大输出 token 数
            max_attempts: 最大重试次数
            retry_delay: 重试间隔（秒）
            error_context: 错误上下文信息

        Returns:
            响应文本列表
        """
        pass

    @abstractmethod
    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 5,
        retry_delay: float = 30,
        poll_interval: float = 3,
        error_context: str = "",
    ) -> List[str]:
        """
        图像生成接口

        Args:
            model_name: 图像模型名称
            prompt: 图像描述提示词
            aspect_ratio: 宽高比
            quality: 图像质量/分辨率
            image_urls: 参考图片 URL 列表（用于 image-to-image）
            max_attempts: 最大重试次数
            retry_delay: 重试间隔（秒）
            poll_interval: 轮询间隔（秒）
            error_context: 错误上下文信息

        Returns:
            base64 编码的图像字符串列表
        """
        pass
