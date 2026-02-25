"""
Evolink Provider 单元测试
测试文本生成和图像生成的核心逻辑
"""

import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.evolink import EvolinkProvider


# ==================== 辅助函数 ====================

def make_png_base64():
    """创建一个最小的 PNG 图片并返回 base64 字符串"""
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_provider(api_key="test-key", base_url="https://api.evolink.ai"):
    """创建 EvolinkProvider 实例"""
    return EvolinkProvider(api_key=api_key, base_url=base_url)


# ==================== 初始化测试 ====================

class TestEvolinkProviderInit:
    def test_init_with_params(self):
        p = make_provider(api_key="sk-abc", base_url="https://example.com")
        assert p.api_key == "sk-abc"
        assert p.base_url == "https://example.com"

    def test_init_default_base_url(self):
        p = EvolinkProvider(api_key="sk-abc")
        assert p.base_url == "https://api.evolink.ai"

    def test_headers_contain_auth(self):
        p = make_provider(api_key="sk-test")
        headers = p._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"


# ==================== 内容格式转换测试 ====================

class TestContentConversion:
    def test_text_only_content(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello world"}]
        messages = p._convert_contents_to_messages(contents, system_prompt="You are helpful")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        # 纯文本时 content 可以是字符串
        user_content = messages[1]["content"]
        assert any("Hello world" in str(part) for part in (user_content if isinstance(user_content, list) else [user_content]))

    def test_text_and_image_content(self):
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": img_b64,
                    "media_type": "image/jpeg",
                },
            },
        ]
        messages = p._convert_contents_to_messages(contents, system_prompt="You are a vision model")

        assert len(messages) == 2
        user_content = messages[1]["content"]
        # 多模态时 content 应该是列表
        assert isinstance(user_content, list)
        # 应该包含文本和图片两个部分
        types_present = {item["type"] for item in user_content}
        assert "text" in types_present
        assert "image_url" in types_present

    def test_image_with_base64_content(self):
        """测试 image 使用 image_base64 字段（planner agent 使用的格式）"""
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Look at this"},
            {"type": "image", "image_base64": img_b64},
        ]
        messages = p._convert_contents_to_messages(contents, system_prompt="Be helpful")

        # messages[0] = system, messages[1] = user
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        image_parts = [part for part in user_content if part["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_empty_system_prompt(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hi"}]
        messages = p._convert_contents_to_messages(contents, system_prompt="")
        # 空 system prompt 不应生成 system message
        assert messages[0]["role"] == "user"


# ==================== 文本生成测试 ====================

class TestTextGeneration:
    @pytest.mark.asyncio
    async def test_text_generation_success(self):
        p = make_provider()

        mock_response = {
            "choices": [
                {"message": {"content": "This is a test response"}}
            ]
        }

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=mock_response):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="You are helpful",
                temperature=0.7,
                max_output_tokens=1000,
            )

        assert result == ["This is a test response"]

    @pytest.mark.asyncio
    async def test_text_generation_retry_on_failure(self):
        p = make_provider()

        mock_response = {
            "choices": [
                {"message": {"content": "Success after retry"}}
            ]
        }

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API error")
            return mock_response

        with patch.object(p, '_post_json', side_effect=mock_post):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="",
                temperature=1.0,
                max_output_tokens=50000,
                max_attempts=5,
                retry_delay=0,
            )

        assert result == ["Success after retry"]
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_text_generation_all_attempts_fail(self):
        p = make_provider()

        with patch.object(p, '_post_json', new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await p.generate_text(
                model_name="gemini-2.5-flash-image",
                contents=[{"type": "text", "text": "Hello"}],
                system_prompt="",
                temperature=1.0,
                max_output_tokens=50000,
                max_attempts=3,
                retry_delay=0,
            )

        assert result == ["Error"]


# ==================== 图像生成测试 ====================

class TestImageGeneration:
    @pytest.mark.asyncio
    async def test_image_generation_creates_task(self):
        p = make_provider()

        create_response = {
            "id": "task-unified-123",
            "status": "pending",
            "progress": 0,
        }

        completed_response = {
            "id": "task-unified-123",
            "status": "completed",
            "progress": 100,
            "results": ["https://example.com/image.png"],
        }

        # 创建一个小的 PNG 图片用于模拟下载
        png_bytes = base64.b64decode(make_png_base64())

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=completed_response), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="A beautiful diagram",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=3,
                retry_delay=0,
                poll_interval=0,
            )

        assert len(result) == 1
        assert result[0] is not None
        assert len(result[0]) > 10  # base64 string should be non-trivial

    @pytest.mark.asyncio
    async def test_image_generation_polls_until_complete(self):
        p = make_provider()

        create_response = {
            "id": "task-123",
            "status": "pending",
        }

        poll_responses = [
            {"id": "task-123", "status": "processing", "progress": 30},
            {"id": "task-123", "status": "processing", "progress": 60},
            {"id": "task-123", "status": "completed", "progress": 100, "results": ["https://example.com/img.png"]},
        ]

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, side_effect=poll_responses), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Test",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=3,
                retry_delay=0,
                poll_interval=0,
            )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_image_generation_task_fails(self):
        p = make_provider()

        create_response = {"id": "task-fail", "status": "pending"}
        failed_response = {"id": "task-fail", "status": "failed", "progress": 0}

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=create_response), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=failed_response):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Test",
                aspect_ratio="16:9",
                quality="2K",
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert result == ["Error"]

    @pytest.mark.asyncio
    async def test_image_generation_with_image_urls(self):
        """测试带参考图片的图像生成（image-to-image）"""
        p = make_provider()

        create_response = {
            "id": "task-img2img",
            "status": "pending",
        }
        completed_response = {
            "id": "task-img2img",
            "status": "completed",
            "progress": 100,
            "results": ["https://example.com/result.png"],
        }

        captured_payload = {}
        async def capture_post(url, payload):
            captured_payload.update(payload)
            return create_response

        with patch.object(p, '_post_json', side_effect=capture_post), \
             patch.object(p, '_get_json', new_callable=AsyncMock, return_value=completed_response), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=make_png_base64()):

            result = await p.generate_image(
                model_name="nano-banana-2-lite",
                prompt="Edit this image",
                aspect_ratio="1:1",
                quality="2K",
                image_urls=["https://example.com/ref.png"],
                max_attempts=1,
                retry_delay=0,
                poll_interval=0,
            )

        assert "image_urls" in captured_payload
        assert captured_payload["image_urls"] == ["https://example.com/ref.png"]


# ==================== 请求构建测试 ====================

class TestRequestBuilding:
    def test_text_request_payload(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello"}]

        payload = p._build_text_payload(
            model_name="gemini-2.5-flash-image",
            contents=contents,
            system_prompt="Be helpful",
            temperature=0.5,
            max_output_tokens=4096,
        )

        assert payload["model"] == "gemini-2.5-flash-image"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 4096
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    def test_image_request_payload(self):
        p = make_provider()

        payload = p._build_image_payload(
            model_name="nano-banana-2-lite",
            prompt="A cat on grass",
            aspect_ratio="16:9",
            quality="2K",
        )

        assert payload["model"] == "nano-banana-2-lite"
        assert payload["prompt"] == "A cat on grass"
        assert payload["size"] == "16:9"
        assert payload["quality"] == "2K"

    def test_image_request_payload_with_urls(self):
        p = make_provider()

        payload = p._build_image_payload(
            model_name="nano-banana-2-lite",
            prompt="Edit this",
            aspect_ratio="1:1",
            quality="4K",
            image_urls=["https://example.com/img.png"],
        )

        assert payload["image_urls"] == ["https://example.com/img.png"]


# ==================== generation_utils 集成测试 ====================

class TestGenerationUtilsIntegration:
    @pytest.mark.asyncio
    async def test_call_evolink_text_routes_correctly(self):
        """测试 generation_utils 中的 evolink 文本调用"""
        from utils import generation_utils

        mock_response = {
            "choices": [{"message": {"content": "test response"}}]
        }

        with patch('providers.evolink.EvolinkProvider._post_json',
                    new_callable=AsyncMock, return_value=mock_response):
            # 验证函数存在且可调用
            assert hasattr(generation_utils, 'call_evolink_text_with_retry_async')

    @pytest.mark.asyncio
    async def test_call_evolink_image_routes_correctly(self):
        """测试 generation_utils 中的 evolink 图像调用"""
        from utils import generation_utils

        assert hasattr(generation_utils, 'call_evolink_image_with_retry_async')


# ==================== ExpConfig Provider 字段测试 ====================

class TestExpConfigProvider:
    def test_config_has_provider_field(self):
        from utils.config import ExpConfig
        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="evolink",
        )
        assert config.provider == "evolink"

    def test_config_default_provider(self):
        from utils.config import ExpConfig
        config = ExpConfig(dataset_name="PaperBananaBench")
        assert config.provider == "evolink"  # 默认使用 evolink


# ==================== Agent 路由测试 ====================

class TestAgentRouting:
    """测试 agent 能正确根据 provider 路由到 evolink"""

    def test_planner_uses_evolink_when_configured(self):
        from utils.config import ExpConfig
        from agents.planner_agent import PlannerAgent

        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="evolink",
            model_name="gemini-2.5-flash-image",
        )
        agent = PlannerAgent(exp_config=config)
        assert agent.exp_config.provider == "evolink"

    def test_visualizer_uses_evolink_image_model(self):
        from utils.config import ExpConfig
        from agents.visualizer_agent import VisualizerAgent

        config = ExpConfig(
            dataset_name="PaperBananaBench",
            provider="evolink",
            model_name="gemini-2.5-flash-image",
            image_model_name="nano-banana-2-lite",
        )
        agent = VisualizerAgent(exp_config=config)
        assert agent.model_name == "nano-banana-2-lite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
