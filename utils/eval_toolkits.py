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
Evaluation toolkits for PaperVizAgent
"""

import json_repair
import json
import asyncio
import base64
import re
from google.genai import types

from prompts import diagram_eval_prompts, plot_eval_prompts
from utils.generation_utils import (
    call_gemini_with_retry_async,
    call_claude_with_retry_async,
    call_openai_with_retry_async,
)

# Prompt mapping: task_name -> eval_dim -> system_prompt
PROMPT_MAP = {
    "diagram": {
        "faithfulness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT,
        "conciseness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT,
        "readability": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT,
        "aesthetics": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT,
    },
    "plot": {
        "faithfulness": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT,
        "conciseness": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT,
        "readability": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT,
        "aesthetics": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT,
    },
}

# Task configuration: task_name -> field labels
TASK_CONFIG = {
    "diagram": {
        "visual_intent_label": "Diagram Caption",
        "raw_content_label": "Methodology Section",
        "human_label": "Human-Drawn Diagram (Human)",
        "model_label": "Model-Generated Diagram (Model)",
    },
    "plot": {
        "visual_intent_label": "Visual Intent of the Desired Plot",
        "raw_content_label": "Raw Data",
        "human_label": "Human-Drawn Plot (Human)",
        "model_label": "Model-Generated Plot (Model)",
    },
}


def _try_regex_extract_winner(text: str) -> str | None:
    """Try to extract winner field using regex as a fallback."""
    patterns = [
        r'"winner"\s*:\s*"([^"]+)"',  # Standard JSON: "winner": "value"
        r'\*\*winner\*\*\s*:\s*"([^"]+)"',  # Markdown bold: **winner**: "value" or **winner**:"value"
        r'\*\*winner\*\*\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)',  # Markdown bold without quotes: **winner**: value (capture until comma, newline, or end)
        r'"winner"\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)',  # Mixed format: "winner": value (no quotes on value, capture until comma, newline, or end)
        r'(?:\*\*|")winner(?:\*\*|")\s*:\s*(?:\*\*|")?([A-Za-z][A-Za-z\s]+?)(?:\*\*|"|,|\n|$)',  # Very flexible: any winner marker followed by colon and value
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = value.rstrip('*"').strip()
            return value
    
    return None


def _extract_winner_with_fallback(clean_json: str, eval_dim: str, valid_winners: list[str]) -> str:
    """Try regex extraction and return winner or 'Error'."""
    extracted = _try_regex_extract_winner(clean_json)
    if extracted and extracted in valid_winners:
        print(f"⚠️  {eval_dim}: regex extracted '{extracted}'")
        return extracted
    print(f"⚠️  {eval_dim}: failed to extract valid winner")
    return "Error"


def _determine_tier_outcome(dim1_outcome: str, dim2_outcome: str) -> str:
    """Determine the outcome for a tier given two dimension outcomes."""
    o1, o2 = dim1_outcome.strip(), dim2_outcome.strip()
    
    # Both agree on a clear winner
    if o1 == o2:
        if o1 in ["Both are good", "Both are bad"]:
            return "Tie"
        return o1
    
    # One Model, one neutral (Both are good/bad)
    if (o1 == "Model" and o2 in ["Both are good", "Both are bad"]) or \
       (o2 == "Model" and o1 in ["Both are good", "Both are bad"]):
        return "Model"
    
    # One Human, one neutral (Both are good/bad)
    if (o1 == "Human" and o2 in ["Both are good", "Both are bad"]) or \
       (o2 == "Human" and o1 in ["Both are good", "Both are bad"]):
        return "Human"
    
    # All other cases (conflicting winners, etc.) -> Tie
    return "Tie"


async def _run_single_eval_ref(
    task_name: str,
    eval_dim: str,
    raw_content: str,
    visual_intent: str,
    gt_image_base64: str,
    model_image_base64: str,
    model_name: str
) -> tuple[str, dict]:
    """Run a single evaluation dimension for referenced comparison."""
    # Get the appropriate prompt based on task_name and eval_dim
    sys_prompt = PROMPT_MAP[task_name][eval_dim]

    # Get task-specific labels
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Invalid task name: {task_name}")
    
    config = TASK_CONFIG[task_name]
    
    # Construct input text based on eval dimension
    if eval_dim in ["readability", "aesthetics"]:
        input_text = f"{config['visual_intent_label']}: {visual_intent}\n{config['human_label']}: "
    else:
        input_text = f"{config['raw_content_label']}: {raw_content}\n{config['visual_intent_label']}: {visual_intent}\n{config['human_label']}: "

    # Construct content list
    content_list = [
        {"type": "text", "text": input_text},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": gt_image_base64,
            },
        },
        {"type": "text", "text": f"\n{config['model_label']}: "},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": model_image_base64,
            },
        },
    ]

    valid_winners = ["Human", "Model", "Both are good", "Both are bad"]
    
    try:
        if "gemini" in model_name:
            response_text_list = await call_gemini_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=1,
                    candidate_count=1,
                    max_output_tokens=50000,
                ),
            )
        elif "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            response_text_list = await call_openai_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_completion_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )
        else:
            response_text_list = await call_claude_with_retry_async(
                model_name=model_name,
                contents=content_list,
                config={
                    "system_prompt": sys_prompt,
                    "temperature": 1,
                    "candidate_num": 1,
                    "max_output_tokens": 10000,
                },
                max_attempts=5,
                retry_delay=30,
            )
        clean_json = response_text_list[0].replace("```json", "").replace("```", "").strip()
        res_obj = json_repair.loads(clean_json)
        
        if not isinstance(res_obj, dict):
            res_obj = {
                "comparison_reasoning": clean_json,
                "winner": _extract_winner_with_fallback(clean_json, eval_dim, valid_winners)
            }
        elif "winner" not in res_obj:
            res_obj["winner"] = _extract_winner_with_fallback(clean_json, eval_dim, valid_winners)
            if "comparison_reasoning" not in res_obj:
                res_obj["comparison_reasoning"] = clean_json
        
        return eval_dim, res_obj
    except Exception as e:
        print(f"❌ {eval_dim}: Evaluation failed - {str(e)[:100]}")
        extracted = _try_regex_extract_winner(clean_json) if 'clean_json' in locals() else None
        winner = extracted if (extracted and extracted in valid_winners) else "Error"
        return eval_dim, {"comparison_reasoning": str(e), "winner": winner}


async def get_score_for_image_referenced(
    sample_data: dict, task_name: str = "diagram", model_name: str = "", work_dir = None
) -> dict:
    """Get score for diagram referenced comparison.
    
    Args:
        sample_data: Sample data dictionary
        task_name: Task name (diagram or plot)
        model_name: Model name for evaluation
        work_dir: Work directory path for resolving relative paths (pathlib.Path)
    """
    from pathlib import Path

    raw_content = sample_data["content"]
    visual_intent = sample_data["visual_intent"]
    
    if "path_to_gt_image" not in sample_data:
        print("⚠️  No ground truth image path found. Skipping evaluation.")
        for dim in ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]:
             sample_data[f"{dim}_outcome"] = "N/A - No GT"
        return sample_data

    path_to_gt_image_rel = sample_data["path_to_gt_image"]
    
    # Resolve relative path using work_dir
    if work_dir:
        path_to_gt_image = work_dir / f"data/PaperBananaBench/{task_name}" / path_to_gt_image_rel
    else:
        # Fallback for backward compatibility (assume it's already absolute)
        path_to_gt_image = Path(path_to_gt_image_rel)

    with open(path_to_gt_image, "rb") as f:
        gt_image_base64 = base64.b64encode(f.read()).decode("utf-8")

    eval_image_field = sample_data["eval_image_field"]
    
    # Check if image was successfully generated
    if eval_image_field not in sample_data:
        print(f"⚠️  Image field '{eval_image_field}' not found. Model generation failed - counting as Human win.")
        # Model failed to generate image, Human wins by default
        for dim in ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]:
            sample_data[f"{dim}_reasoning"] = "Model failed to generate image - Human wins by default"
            sample_data[f"{dim}_outcome"] = "Human"
        return sample_data
    
    model_image_base64 = sample_data[eval_image_field]

    # Run evaluations for all dimensions
    dims = ["faithfulness", "conciseness", "readability", "aesthetics"]
    tasks = [
        _run_single_eval_ref(
            task_name,
            dim,
            raw_content,
            visual_intent,
            gt_image_base64,
            model_image_base64,
            model_name
        ) for dim in dims
    ]

    results = await asyncio.gather(*tasks)
    for eval_dim, res_obj in results:
        sample_data[f"{eval_dim}_reasoning"] = res_obj.get("comparison_reasoning", "")
        sample_data[f"{eval_dim}_outcome"] = res_obj.get("winner", "Unknown")

    faithfulness = sample_data.get("faithfulness_outcome", "Unknown")
    readability = sample_data.get("readability_outcome", "Unknown")
    conciseness = sample_data.get("conciseness_outcome", "Unknown")
    aesthetics = sample_data.get("aesthetics_outcome", "Unknown")
    
    # Tier 1: Faithfulness + Readability
    tier1_outcome = _determine_tier_outcome(faithfulness, readability)
    
    if tier1_outcome in ["Model", "Human"]:
        overall_outcome = tier1_outcome
        decision_path = f"Tier1({faithfulness}, {readability}) -> {tier1_outcome} [Decided at Tier 1]"
    else:
        # Tier 1 is tied, check Tier 2
        tier2_outcome = _determine_tier_outcome(conciseness, aesthetics)
        overall_outcome = tier2_outcome
        decision_path = f"Tier1({faithfulness}, {readability}) -> Tie; Tier2({conciseness}, {aesthetics}) -> {tier2_outcome} [Decided at Tier 2]"
    
    sample_data["overall_outcome"] = overall_outcome
    sample_data["overall_reasoning"] = f"Rule-based calculation: {decision_path}"

    return sample_data