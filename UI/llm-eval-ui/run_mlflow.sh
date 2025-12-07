#!/bin/bash
# Start MLflow UI server

echo "🔬 Starting MLflow UI..."
echo ""

# Go to project root (where mlruns/ directory is)
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator

echo "✓ Working directory: $(pwd)"
echo "✓ MLflow tracking: mlruns/"
echo ""
echo "Starting MLflow UI on port 5000..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run mlflow using full path from llm311 env
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/mlflow ui --port 5000
