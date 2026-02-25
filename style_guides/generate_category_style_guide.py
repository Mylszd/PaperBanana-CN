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

import asyncio
import os
import json
import pathlib
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# ================= 1. Configuration =================
MODE = "diagram"

# Define work directory and data directory for path resolution
WORK_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = WORK_DIR / f"data/PaperBananaBench/{MODE}"

INPUT_JSON_PATH = WORK_DIR / f"data/PaperBananaBench/{MODE}/ref.json"
OUTPUT_REPORT_PATH = f"style_guides/neurips2025_{MODE}_style_guide.md"
BATCH_OUTPUT_DIR = "tmp/style_analysis_all"

# Sampling settings
NUM_SAMPLES = 50  # None for all, or a number to limit

MODEL_NAME = "gemini-3-pro-preview"

# Batch processing settings
BATCH_SIZE = 20
CONCURRENCY_LIMIT = 5

# ================= 2. Initialize Client =================
client = genai.Client()

# ================= 3. Improved Prompt Design =================


DIAGRAM_BATCH_ANALYSIS_PROMPT = """
You are a Lead Information Designer analyzing the visual style of top-tier AI conference papers (NeurIPS 2025).
I have attached a batch of methodology diagrams from the NeurIPS 2025 conference.

**Your Task:**
Summarize a visual design guideline that ignores the specific scientific algorithms. Focus ONLY on the **Aesthetic and Graphic Design** choices.

**Critical:** Do NOT converge each element to a single fixed design choice. Instead, identify what common design choices exist for each element and which ones are more popular or preferred.

Please focus on these specific dimensions:
1.  **Color Palette:** Observe color schemes, saturation levels, etc. Notice aesthetically pleasing combinations and preserve multiple options.
2.  **Shapes & Containers:** Observe shape choices (e.g., rounded vs. sharp rectangles), containers, borders (thickness, color), background fills, shadows, etc.
3.  **Lines & Arrows:** Observe line thickness, colors, arrow styles, dashed line usage.
4.  **Layout & Composition:** Observe layouts, element arrangement patterns, information density, whitespace usage.
5.  **Typography & Icons:** Observe font weights, sizes, colors, usage patterns, and icon usage.

Please note that papers of different domains may have different aesthetic preferences. For example, agent papers will use detailed, cartoon-like illustrative styles more often, while theorectical papers will use more minimalistic styles. When you are summarizing the style, please consider the domain of the paper. You can use "For [domain], common options include: [list]" format to describe the style.

Return a concise bullet-point summary of the visual style diversity observed in this batch.
"""

PLOT_BATCH_ANALYSIS_PROMPT = """
You are a Lead Information Designer analyzing the visual style of top-tier AI conference papers (NeurIPS 2025).
I have attached a batch of statistical plots from the NeurIPS 2025 conference.

**Your Task:**
Summarize a visual design guideline for statistical plots. Focus ONLY on the **Aesthetic and Graphic Design** choices (not the data itself).

**Critical:** Do NOT converge each element to a single fixed design choice. Instead, identify what common design choices exist for each element and which ones are more popular or preferred.

Please focus on these specific dimensions:
1.  **Color Palette:** Observe color schemes for categorical data, sequential gradients for heatmaps, and diverging scales. Identify aesthetically pleasing combinations.
2.  **Axes & Grids:** Observe the styling of x/y axes, tick marks, and grid lines (e.g., light gray, dashed, none). Note the line weights and colors.
3.  **Data Representation (by Type):**
    *   **Bar Chart:** Bar width, spacing, borders, and error bar styles.
    *   **Line Chart:** Line thickness, transparency, marker styles (circles, squares, etc.), and shadow/area fills.
    *   **Tree & Pie Chart:** Node shapes, edge styles, and slice explosion/labeling.
    *   **Scatter Plot:** Marker transparency (alpha), size, and overlap handling.
    *   **Heatmap:** Colormap choices (e.g., Viridis, Magma, custom), cell borders, and aspect ratios.
    *   **Radar Chart:** Grid structure, polygon fill transparency, and axis labeling.
    *   **Miscellaneous:** Observe styles for other specialized types.
4.  **Layout & Composition:** Legend placement, whitespace balance, margins, and subplot arrangements.
5.  **Typography:** Font weights, sizes, and colors for titles, axis labels, and annotations.

Return a concise bullet-point summary of the visual style diversity observed for these plot types in this batch.
"""


PLOT_FINAL_SUMMARY_PROMPT = """
Below are multiple visual analysis reports from a dataset of NeurIPS 2025 statistical plots.
Your goal is to synthesize these into a **"NeurIPS 2025 Statistical Plot Aesthetics Guide"**.

**Target Audience:** A researcher who wants to create plots that look "professional" and "NeurIPS-style".

**Critical Philosophy:** This is NOT about prescribing a single "correct" design. Instead, summarize the **multiple accepted design choices** in this field.

**Output Structure:**
1.  **The "NeurIPS Look" for Plots:** A high-level description of the prevailing aesthetic vibe (e.g., minimalistic, high-contrast, specific color schemes).
2.  **Detailed Style Options:**
    * **Color Palettes:** Common color sets for different data types (categorical, sequential).
    * **Axes & Grids:** Prevailing conventions for grid visibility and axis styling.
    * **Layout & Typography:** Common legend positions and font preferences.
3.  **Type-Specific Guidelines:**
    * Summarize specific aesthetic preferences for: *Bar Chart*, *Line Chart*, *Tree & Pie Chart*, *Scatter Plot*, *Heatmap*, *Radar Chart*, and *Miscellaneous*.
4.  **Common Pitfalls:** What design choices make a plot look "amateur" or "outdated" (e.g., default Excel/old Matplotlib styles)?

**Formatting Guidelines:**
- Use "Common choices include: [Option A], [Option B]" format.
- Frame everything as OBSERVATIONS not PRESCRIPTIONS.
- Focus on aesthetic quality and professional rendering.

**Input Reports:**
{all_reports}
"""



DIAGRAM_FINAL_SUMMARY_PROMPT = """
Below are multiple visual analysis reports from a dataset of NeurIPS 2025 method diagrams.
Your goal is to synthesize these into a **"NeurIPS 2025 Method Diagram Aesthetics Guide"**.

**Target Audience:** A researcher who wants to draw a diagram that looks "professional" and "accepted" by the community.

**Critical Philosophy:** This is NOT about prescribing a single "correct" design. Instead, summarize the **multiple accepted design choices** in this field.

**AVOID These Anti-Patterns:**
1. **DO NOT create rigid semantic bindings** like "Light Blue is standard for encoders" or "LLMs use brain icons". 
2. **DO NOT prescribe icon-to-concept mappings** like "🧠 Brain (LLM/Reasoning Core)".
3. **Present COLOR as aesthetic OPTIONS, not functional rules**.
   - Focus on: "These color combinations look good together" rather than "This component type requires this color"

**Output Structure:**
1.  **The "NeurIPS Look":** A high-level description of the prevailing aesthetic vibe.
2.  **Detailed Style Options:**
    * **Colors:** What aesthetically pleasing color palettes are common? List hex codes and describe combinations, NOT what component types they're "for".
    * **Shapes & Containers:** Common shape choices, border styles, shadow usage patterns.
    * **Lines & Arrows:** Common line styles, arrow types, and dashed line conventions.
    * **Layout & Composition:** Common layout patterns and information density preferences.
    * **Typography & Icons:** Common font choices. For icons: describe what icon OPTIONS are available for different purposes (format: "For [purpose], common options include: [icon1], [icon2]...")
3.  **Common Pitfalls:** What design choices make a diagram look "outdated" or "amateur"?
4.  **Domain-Specific Styles:** What are the common styles used in different domains? For example, agent papers will use detailed, cartoon-like illustrative styles more often, while theorectical papers will use more minimalistic styles.

**Formatting Guidelines for Options:**
- If 80%+ prevalence: "Most papers use [Option A]..."
- If multiple popular options: "Common choices include: [Option A] (~X%), [Option B] (~Y%)..."
- For icons/colors: Use "For representing [concept], observed options include: [list]" format
- Frame everything as OBSERVATIONS not PRESCRIPTIONS
- Emphasize aesthetic quality over semantic rules

**Input Reports:**
{all_reports}
"""





# Selective Prompt Selection
if MODE == "plot":
    BATCH_ANALYSIS_PROMPT = PLOT_BATCH_ANALYSIS_PROMPT
    FINAL_SUMMARY_PROMPT = PLOT_FINAL_SUMMARY_PROMPT
else:
    BATCH_ANALYSIS_PROMPT = DIAGRAM_BATCH_ANALYSIS_PROMPT
    FINAL_SUMMARY_PROMPT = DIAGRAM_FINAL_SUMMARY_PROMPT


# ================= 4. Core Logic =================

async def analyze_batch(sem, batch_index, image_paths):
    """Analyze a batch of images for visual patterns"""
    async with sem:
        valid_parts = []
        loaded_count = 0
        
        # Load images
        for p in image_paths:
            try:
                path_obj = pathlib.Path(p)
                if path_obj.exists():
                    file_bytes = path_obj.read_bytes()
                    mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
                    valid_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime))
                    loaded_count += 1
            except Exception as e:
                print(f"Error reading {p}: {e}")
        
        if not valid_parts:
            return f"[Batch {batch_index}] Skipped: No valid images found."
        
        # Call Gemini for visual analysis
        contents = valid_parts + [BATCH_ANALYSIS_PROMPT]
        
        try:
            response = await client.aio.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=1,
                    max_output_tokens=50000
                ),
            )
            
            report_content = response.text
            
            # Save intermediate results
            output_filename = os.path.join(BATCH_OUTPUT_DIR, f"batch_{batch_index}.txt")
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            return f"=== Batch {batch_index} Analysis ({loaded_count} images) ===\n{report_content}\n"
        
        except Exception as e:
            error_msg = f"Error in Batch {batch_index}: {str(e)}"
            print(error_msg)
            return f"=== {error_msg} ===\n"

async def main_task():
    """Main async task"""
    # Step 0: Prepare output directory
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    print(f"📁 Output directory: {BATCH_OUTPUT_DIR}")
    
    # Step 1: Load and filter data
    print(f"📂 Loading {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract image paths and resolve relative paths
    all_image_paths = []
    for item in data:
        path_rel = item.get('path_to_gt_image')
        if path_rel:
            # Resolve relative path using DATA_DIR
            path = DATA_DIR / path_rel
            if path.exists():
                all_image_paths.append(str(path))
    
    print(f"📸 Found {len(all_image_paths)} valid image paths")
    
    # Apply sampling limit
    if NUM_SAMPLES is not None:
        all_image_paths = all_image_paths[:NUM_SAMPLES]
        print(f"✂️  Limiting to first {NUM_SAMPLES} samples")
    
    if not all_image_paths:
        print("❌ No valid images found")
        return
    
    # Step 2: Split into batches
    batches = [all_image_paths[i:i + BATCH_SIZE] 
               for i in range(0, len(all_image_paths), BATCH_SIZE)]
    print(f"📦 Split into {len(batches)} batches (size: {BATCH_SIZE})")
    
    # Step 3: Parallel analysis
    print(f"🚀 Starting visual analysis...")
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [analyze_batch(sem, i, b) for i, b in enumerate(batches)]
    
    batch_reports = await tqdm.gather(*tasks, desc="Analyzing Batches")
    
    # Step 4: Synthesize final style guide
    print("\n🧠 Synthesizing final style guide...")
    combined_text = "\n\n".join(batch_reports)
    
    try:
        final_response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=[FINAL_SUMMARY_PROMPT.format(
                num_batches=len(batches),
                all_reports=combined_text
            )],
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=16000
            ),
        )
        
        # Save final style guide
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(final_response.text)
        
        print(f"\n🎉 Success! Style guide saved to: {OUTPUT_REPORT_PATH}")
        print("="*80)
        print("Preview:")
        print("="*80)
        print(final_response.text[:800] + "...\n")
        print(f"(See full content in {OUTPUT_REPORT_PATH})")
        
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")

# ================= 5. Entry Point =================
if __name__ == "__main__":
    asyncio.run(main_task())
