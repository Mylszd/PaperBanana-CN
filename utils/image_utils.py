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
Image utility functions for processing and converting images
"""

import base64
import io
from PIL import Image


def convert_png_b64_to_jpg_b64(png_b64_str: str) -> str:
    """
    Convert a PNG base64 string to a JPG base64 string.
    
    Args:
        png_b64_str: Base64 encoded PNG image string
        
    Returns:
        Base64 encoded JPG image string, or None if conversion fails
    """
    try:
        if not png_b64_str or len(png_b64_str) < 10:
            print(f"⚠️  Invalid base64 string (too short): {png_b64_str[:50] if png_b64_str else 'None'}")
            return None
            
        img = Image.open(io.BytesIO(base64.b64decode(png_b64_str))).convert("RGB")
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", quality=95)
        return base64.b64encode(out_io.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        print(f"   Input preview: {png_b64_str[:100] if png_b64_str else 'None'}")
        return None
