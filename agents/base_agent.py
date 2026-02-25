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
Base class for agents
"""

import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import json_repair

from utils.config import ExpConfig


class BaseAgent(ABC):
    """Base class for agents"""

    def __init__(
        self,
        model_name: str = "",
        system_prompt: str = "",
        exp_config: "ExpConfig" = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.exp_config = exp_config

    @abstractmethod
    async def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process the input data and return the result.
        
        Args:
            data: Input data dictionary
            **kwargs: Additional subclass-specific parameters
        
        Returns:
            Processed data dictionary
        """
