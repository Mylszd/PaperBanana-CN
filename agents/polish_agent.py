# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Polish Agent - 根据风格指南优化基准图像。
"""

import base64
import io
from pathlib import Path
from typing import Dict, Any
from PIL import Image

from utils import generation_utils, image_utils
from .base_agent import BaseAgent


def _load_image_as_base64(image_path: str) -> str:
    """Load an image from path and convert to base64"""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None


class PolishAgent(BaseAgent):
    """Polish Agent to apply style guidelines to ground truth images"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_model_name = self.exp_config.image_model_name
        self.text_model_name = self.exp_config.model_name

        if self.exp_config.task_name == "plot":
            self.style_guide_filename = "neurips2025_plot_style_guide.md"
            self.suggestion_system_prompt = PLOT_SUGGESTION_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
            }
        else:
            self.style_guide_filename = "neurips2025_diagram_style_guide.md"
            self.suggestion_system_prompt = DIAGRAM_SUGGESTION_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
            }

    async def _generate_suggestions(self, gt_image_b64: str, style_guide: str) -> str:
        """Step 1: Generate improvement suggestions based on style guide"""
        user_prompt = f"Here is the style guide:\n{style_guide}\n\nPlease analyze the provided image against this style guide and list up to 10 specific improvement suggestions to make the image visually more appealing. If the image is already perfect, just say 'No changes needed'."

        content_list = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": gt_image_b64
                }
            }
        ]

        try:
            if self.exp_config.provider == "evolink":
                response_list = await generation_utils.call_evolink_text_with_retry_async(
                    model_name=self.text_model_name,
                    contents=content_list,
                    config={
                        "system_prompt": self.suggestion_system_prompt,
                        "temperature": 1,
                        "max_output_tokens": 50000,
                    },
                    max_attempts=3,
                    retry_delay=10,
                )
            else:
                from google.genai import types
                response_list = await generation_utils.call_gemini_with_retry_async(
                    model_name=self.text_model_name,
                    contents=content_list,
                    config=types.GenerateContentConfig(
                        system_instruction=self.suggestion_system_prompt,
                        temperature=1,
                        candidate_count=1,
                        max_output_tokens=50000,
                    ),
                    max_attempts=3,
                    retry_delay=10,
                )
            return response_list[0] if response_list else ""
        except Exception as e:
            print(f"❌ Error during suggestion generation: {e}")
            return ""

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        task_name = cfg["task_name"]

        gt_image_path_rel = data.get("path_to_gt_image")
        if not gt_image_path_rel:
            print(f"⚠️  No GT image path found in data")
            return data

        gt_image_path = self.exp_config.work_dir / f"data/PaperBananaBench/{task_name}" / gt_image_path_rel

        gt_image_b64 = _load_image_as_base64(str(gt_image_path))
        if not gt_image_b64:
            print(f"⚠️  Failed to load GT image from {gt_image_path}")
            return data

        style_guide_path = self.exp_config.work_dir / "style_guides" / self.style_guide_filename
        try:
            with open(style_guide_path, "r", encoding="utf-8") as f:
                style_guide = f.read()
        except Exception as e:
            print(f"❌ Error loading style guide from {style_guide_path}: {e}")
            return data

        print(f"🎨 [Step 1] Generating suggestions for {task_name}...")
        suggestions = await self._generate_suggestions(gt_image_b64, style_guide)

        if not suggestions or "No changes needed" in suggestions:
            print(f"✨ No changes needed for this image.")
            pass

        if suggestions:
            data[f"suggestions_{task_name}"] = suggestions

        print(f"📝 Suggestions: {suggestions[:200]}...")

        # Step 2: Polish Image using suggestions
        print(f"🎨 [Step 2] Polishing image with suggestions...")
        user_prompt = f"Please polish this image based on the following suggestions:\n\n{suggestions}\n\nPolished Image:"

        content_list = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": gt_image_b64
                }
            }
        ]

        try:
            if self.exp_config.provider == "evolink":
                # Evolink 图像生成：先上传参考图获取 URL，再传给 image_urls
                print(f"🎨 [Step 2a] 上传参考图到 Evolink 文件服务...")
                ref_image_url = await generation_utils.upload_image_to_evolink(
                    gt_image_b64, media_type="image/jpeg"
                )
                response_list = await generation_utils.call_evolink_image_with_retry_async(
                    model_name=self.image_model_name,
                    prompt=user_prompt,
                    config={
                        "aspect_ratio": data.get("additional_info", {}).get("rounded_ratio", "16:9"),
                        "quality": "2K",
                        "image_urls": [ref_image_url],
                    },
                    max_attempts=5,
                    retry_delay=30,
                )
            else:
                from google.genai import types
                response_list = await generation_utils.call_gemini_with_retry_async(
                    model_name=self.image_model_name,
                    contents=content_list,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=self.exp_config.temperature,
                        candidate_count=1,
                        max_output_tokens=50000,
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=data.get("additional_info", {}).get("rounded_ratio", "16:9"),
                            image_size="1k",
                        ),
                    ),
                    max_attempts=5,
                    retry_delay=30,
                )

            if response_list and response_list[0]:
                converted_jpg = image_utils.convert_png_b64_to_jpg_b64(response_list[0])
                if converted_jpg:
                    output_key = f"polished_{task_name}_base64_jpg"
                    data[output_key] = converted_jpg
                else:
                    print(f"⚠️  Image conversion failed")
            else:
                print(f"⚠️  No response from model")

        except Exception as e:
            print(f"❌ Error during image generation: {e}")

        return data


DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".
"""

PLOT_SUGGESTION_SYSTEM_PROMPT = """
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".
"""

DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions.
"""

PLOT_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information in the plot.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions (colors, fonts, layout, etc.).
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards for top-tier conferences.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions.
"""
