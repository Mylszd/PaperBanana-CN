#!/bin/bash
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

set -e

# Resolve Project Root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
echo "Installing/checking dependencies..."
uv pip install -r requirements.txt

# Create necessary data directories if they don't exist
mkdir -p data/PaperBananaBench/diagram
mkdir -p data/PaperBananaBench/plot
if [ ! -f "data/PaperBananaBench/diagram/ref.json" ]; then
    echo "[]" > data/PaperBananaBench/diagram/ref.json
fi
if [ ! -f "data/PaperBananaBench/plot/ref.json" ]; then
    echo "[]" > data/PaperBananaBench/plot/ref.json
fi

# Run Streamlit
echo "Starting Streamlit..."
streamlit run demo.py --server.port 8501 --server.address 0.0.0.0
