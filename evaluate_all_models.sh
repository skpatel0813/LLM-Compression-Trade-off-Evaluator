#!/bin/bash
# evaluate_all_models.sh - Comprehensive evaluation of all models

set -e

echo "======================================================================"
echo "Comprehensive Model Evaluation"
echo "======================================================================"
echo ""
echo "Models to evaluate:"
echo "  1. Baseline 8B (meta-llama/Meta-Llama-3.1-8B-Instruct)"
echo "  2. KD-trained on MBPP only (outputs/llama31_8b_mbpp_kd_full)"
echo "  3. KD-trained on Combined dataset (outputs/llama31_8b_combined_kd_lora)"
echo ""
echo "Benchmarks:"
echo "  - Combined test set (10% of MBPP+HumanEval)"
echo "  - HumanEval (164 problems)"
echo "  - MBPP test set (257 problems)"
echo ""
echo "======================================================================"

# Set environment
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

RESULTS_DIR="results/comprehensive_eval"
mkdir -p $RESULTS_DIR

# ============================================================================
# 1. Evaluate on Combined Test Set (10% split)
# ============================================================================
echo ""
echo "[1/3] Evaluating on Combined Test Set..."
echo ""

# Baseline 8B
echo "  [1.1] Baseline 8B on combined test..."
python evaluate_combined.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output $RESULTS_DIR/baseline_8b_combined_test.jsonl \
    --bf16

# MBPP-only KD
if [ -d "outputs/llama31_8b_mbpp_kd_full" ]; then
    echo "  [1.2] MBPP-KD model on combined test..."
    python evaluate_combined.py \
        --model outputs/llama31_8b_mbpp_kd_full \
        --output $RESULTS_DIR/mbpp_kd_combined_test.jsonl \
        --bf16
fi

# Combined KD
if [ -d "outputs/llama31_8b_combined_kd_lora" ]; then
    echo "  [1.3] Combined-KD model on combined test..."
    python evaluate_combined.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --lora_dir outputs/llama31_8b_combined_kd_lora \
        --output $RESULTS_DIR/combined_kd_combined_test.jsonl \
        --bf16
fi

# ============================================================================
# 2. Evaluate on HumanEval
# ============================================================================
echo ""
echo "[2/3] Evaluating on HumanEval..."
echo ""

# Baseline 8B
echo "  [2.1] Baseline 8B on HumanEval..."
python src/eval_humaneval_mbpp_style.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output $RESULTS_DIR/baseline_8b_humaneval.jsonl \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --bf16

# MBPP-only KD
if [ -d "outputs/llama31_8b_mbpp_kd_full" ]; then
    echo "  [2.2] MBPP-KD model on HumanEval..."
    python src/eval_humaneval_mbpp_style.py \
        --model outputs/llama31_8b_mbpp_kd_full \
        --output $RESULTS_DIR/mbpp_kd_humaneval.jsonl \
        --max_new_tokens 512 \
        --temperature 0.2 \
        --bf16
fi

# Combined KD
if [ -d "outputs/llama31_8b_combined_kd_lora" ]; then
    echo "  [2.3] Combined-KD model on HumanEval..."
    python src/eval_humaneval_mbpp_style.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --lora_dir outputs/llama31_8b_combined_kd_lora \
        --output $RESULTS_DIR/combined_kd_humaneval.jsonl \
        --max_new_tokens 512 \
        --temperature 0.2 \
        --bf16
fi

# ============================================================================
# 3. Evaluate on MBPP Test Set
# ============================================================================
echo ""
echo "[3/3] Evaluating on MBPP Test Set..."
echo ""

# Baseline 8B
echo "  [3.1] Baseline 8B on MBPP..."
python evaluate_combined.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --test_file data/mbpp_test.jsonl \
    --output $RESULTS_DIR/baseline_8b_mbpp_test.jsonl \
    --bf16

# MBPP-only KD
if [ -d "outputs/llama31_8b_mbpp_kd_full" ]; then
    echo "  [3.2] MBPP-KD model on MBPP..."
    python evaluate_combined.py \
        --model outputs/llama31_8b_mbpp_kd_full \
        --test_file data/mbpp_test.jsonl \
        --output $RESULTS_DIR/mbpp_kd_mbpp_test.jsonl \
        --bf16
fi

# Combined KD
if [ -d "outputs/llama31_8b_combined_kd_lora" ]; then
    echo "  [3.3] Combined-KD model on MBPP..."
    python evaluate_combined.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --lora_dir outputs/llama31_8b_combined_kd_lora \
        --test_file data/mbpp_test.jsonl \
        --output $RESULTS_DIR/combined_kd_mbpp_test.jsonl \
        --bf16
fi

# ============================================================================
# 4. Summarize Results
# ============================================================================
echo ""
echo "======================================================================"
echo "Generating Summary Report..."
echo "======================================================================"

python << 'EOF'
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path("results/comprehensive_eval")
metrics = defaultdict(dict)

# Collect all metrics
for metrics_file in results_dir.glob("*.metrics.json"):
    with open(metrics_file) as f:
        data = json.load(f)
        
        # Extract model name and benchmark
        filename = metrics_file.stem.replace('.metrics', '')
        parts = filename.rsplit('_', 1)
        
        if len(parts) == 2:
            model_name = parts[0]
            benchmark = parts[1]
        else:
            model_name = filename
            benchmark = "unknown"
        
        # Store metrics
        if "overall_accuracy" in data:
            metrics[model_name][benchmark] = data["overall_accuracy"]
        elif "metrics" in data and "pass@1" in data["metrics"]:
            metrics[model_name][benchmark] = data["metrics"]["pass@1"]

# Print summary table
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION RESULTS")
print("="*80)
print(f"\n{'Model':<40} {'Combined':<12} {'HumanEval':<12} {'MBPP':<12}")
print("-"*80)

model_order = ["baseline_8b", "mbpp_kd", "combined_kd"]
benchmark_order = ["test", "humaneval", "test"]  # combined_test, humaneval, mbpp_test

for model in model_order:
    row = f"{model:<40}"
    
    # Combined test
    combined_val = metrics[model].get("test", metrics[model].get("combined", 0))
    row += f" {combined_val*100:>6.2f}%    "
    
    # HumanEval
    he_val = metrics[model].get("humaneval", 0)
    row += f" {he_val*100:>6.2f}%    "
    
    # MBPP test
    mbpp_val = metrics[model].get("test", 0) if model != "mbpp_kd" else metrics[model].get("mbpp", 0)
    row += f" {mbpp_val*100:>6.2f}%"
    
    print(row)

print("="*80)

# Save summary
summary = {
    "models": list(metrics.keys()),
    "benchmarks": ["combined_test", "humaneval", "mbpp_test"],
    "results": dict(metrics)
}

with open("results/comprehensive_eval/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✓ Summary saved to: results/comprehensive_eval/summary.json")
print("="*80 + "\n")
EOF

echo ""
echo "======================================================================"
echo "✅ All evaluations complete!"
echo "======================================================================"
echo ""
echo "Results location: $RESULTS_DIR/"
echo ""
echo "To view summary:"
echo "  cat $RESULTS_DIR/summary.json"
echo ""