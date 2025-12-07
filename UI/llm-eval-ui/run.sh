#!/bin/bash
# Quick start script for LLM Eval UI - Streamlit App

echo "🚀 Starting LLM Evaluation Dashboard..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "✓ Working directory: $(pwd)"
echo ""
echo "Starting Streamlit app..."
echo "Access at: http://localhost:8502"
echo ""
echo "Note: Make sure conda environment 'llm311' is activated!"
echo "      Run: conda activate llm311"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit using full path from llm311 env
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/streamlit run app.py --server.port 8502
