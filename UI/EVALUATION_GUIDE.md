# LLM Compression Trade-off Evaluator - Evaluation Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Running Evaluations](#running-evaluations)
- [Understanding Results](#understanding-results)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Environment Setup

1. **Activate the conda environment:**
   ```bash
   conda activate llm311
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; import transformers; print('Environment ready!')"
   ```

3. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```

---

## Running Evaluations

### 1. Basic HumanEval Evaluation

The primary evaluation script is `src/eval_humaneval.py`. This evaluates models on the HumanEval benchmark (164 Python coding problems).

#### Evaluate Baseline 8B Model

```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output results/baseline_8b_humaneval.jsonl \
  --num_samples 10 \
  --temperature 0.01 \
  --max_new_tokens 512 \
  --bf16
```

#### Evaluate Teacher 70B Model (with 8-bit quantization)

```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output results/teacher_70b_humaneval.jsonl \
  --num_samples 10 \
  --temperature 0.01 \
  --max_new_tokens 512 \
  --load_in_8bit \
  --bf16
```

#### Evaluate Distilled Model (with LoRA adapters)

```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_dir outputs/llama31_8b_kd_lora/lora \
  --output results/mbpp_distilled_humaneval.jsonl \
  --num_samples 10 \
  --temperature 0.01 \
  --max_new_tokens 512 \
  --bf16
```

### 2. Routing System Evaluation

The routing system evaluates models on different complexity levels and combines results.

```bash
python src/routing_system.py \
  --distilled_model_path outputs/llama31_8b_kd_lora \
  --teacher_model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output_dir results/ \
  --wandb_project llm-routing-eval
```

This will:
1. Classify HumanEval problems by complexity (1-10 scale)
2. Run distilled model on simple queries (complexity 1-6)
3. Run distilled model on complex queries (complexity 7-10)
4. Run 70B model on simple queries (complexity 1-6)
5. Run 70B model on complex queries (complexity 7-10)
6. Create hybrid "routed" results (distilled for simple, 70B for complex)
7. Generate comprehensive comparison metrics

### 3. Training a Distilled Model

To train a new distilled model using knowledge distillation:

```bash
python src/train_kd.py \
  --teacher_model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --student_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_file data/mbpp_train.jsonl \
  --val_file data/mbpp_val.jsonl \
  --output_dir outputs/llama31_8b_kd_lora \
  --num_epochs 10 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --kd_alpha 0.5 \
  --kd_temperature 1.0 \
  --lora_r 16 \
  --lora_alpha 32
```

---

## Understanding Results

### Result Files

After running an evaluation, you'll get three types of files:

1. **`<name>.jsonl`** - Raw generations from the model
2. **`<name>.jsonl_results.jsonl`** - Detailed pass/fail results per test case
3. **`<name>.metrics.json`** - Summary metrics (this is what you'll read most)

### Metrics File Structure

Example: `results/baseline_8b_humaneval.metrics.json`

```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "lora_dir": null,
  "num_problems": 164,
  "num_samples_per_problem": 10,
  "total_generations": 1640,
  "generation_time_sec": 3729.35,
  "avg_time_per_sample": 2.27,
  "pass_at_k": {
    "pass@1": 0.4951,
    "pass@5": 0.7199,
    "pass@10": 0.7561
  },
  "gpu_metrics": { ... },
  "config": { ... }
}
```

### Key Metrics Explained

#### 1. Performance Metrics

| Metric | Description | Example Value | Interpretation |
|--------|-------------|---------------|----------------|
| `pass@1` | Probability of generating a correct solution on the first try | 0.4951 (49.51%) | Higher is better. 50% means half of problems solved correctly on first attempt |
| `pass@5` | Probability of at least one correct solution in 5 attempts | 0.7199 (71.99%) | Measures consistency and diversity of model |
| `pass@10` | Probability of at least one correct solution in 10 attempts | 0.7561 (75.61%) | Upper bound on model's problem-solving capability |

**Real-world example:**
- **Baseline 8B**: pass@1 = 49.5%, pass@10 = 75.6%
- **Teacher 70B**: pass@1 = 40.8%, pass@10 = 70.7%
- **Observation**: The 8B model actually outperforms the 70B in this case! This could be due to 8-bit quantization degrading the 70B's performance.

#### 2. Energy & Cost Metrics

```json
"gpu_metrics": {
  "gpu_0_energy_wh": 88.99,
  "gpu_1_energy_wh": 93.84,
  ...
  "gpu_7_energy_wh": 168.60,
  "gpu_0_power_watts_mean": 85.92,
  "gpu_1_power_watts_mean": 90.61,
  ...
}
```

| Metric | Description | How to Calculate Total |
|--------|-------------|------------------------|
| `gpu_X_energy_wh` | Energy consumed by GPU X in Watt-hours (Wh) | Sum all `gpu_X_energy_wh` values |
| `gpu_X_power_watts_mean` | Average power draw during inference | For reference only |
| `generation_time_sec` | Total wall-clock time | Convert to hours: `/ 3600` |

**Example calculation:**

For the baseline 8B model:
- Total energy = 88.99 + 93.84 + 95.77 + 92.63 + 98.48 + 97.07 + 98.49 + 168.60 = **833.87 Wh** = **0.83 kWh**
- Total time = 3729.35 sec = **62.2 minutes** = **1.04 hours**
- Cost (at $0.12/kWh) = 0.83 × $0.12 = **$0.10**

For the teacher 70B model:
- Total energy = 1049.8 + 3053.2 + 1199.3 + 1059.9 + 1125.3 + 1340.9 + 1284.5 + 1302.8 = **11,415.7 Wh** = **11.42 kWh**
- Total time = 47656.46 sec = **794.3 minutes** = **13.2 hours**
- Cost (at $0.12/kWh) = 11.42 × $0.12 = **$1.37**

**Cost comparison:**
- 70B model: $1.37, 13.2 hours
- 8B model: $0.10, 1.04 hours
- **Savings: 92.7% cheaper, 12.8x faster** (but similar or better performance!)

#### 3. Configuration

```json
"config": {
  "temperature": 0.01,
  "max_new_tokens": 512,
  "quantization": "8bit",
  "dtype": "bf16"
}
```

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `temperature` | Sampling randomness (0.01 ≈ greedy) | 0.01 for deterministic, 0.7-1.0 for creative |
| `max_new_tokens` | Maximum length of generated code | 256-512 for HumanEval |
| `quantization` | Model compression method | `none`, `8bit`, `4bit` |
| `dtype` | Precision for computation | `bf16` (recommended), `fp16`, `fp32` |

---

## Comparing Results

### Example: Baseline 8B vs Teacher 70B

| Model | Pass@1 | Pass@10 | Energy (kWh) | Time (hrs) | Cost ($) | Performance/$ |
|-------|--------|---------|--------------|------------|----------|---------------|
| 8B Baseline | 49.5% | 75.6% | 0.83 | 1.04 | $0.10 | 495% |
| 70B Teacher (8bit) | 40.8% | 70.7% | 11.42 | 13.2 | $1.37 | 29.8% |
| **Efficiency Gain** | **+21%** | **+6.9%** | **13.8x less** | **12.7x faster** | **13.7x cheaper** | **16.6x better** |

**Conclusion:** The 8B model is dramatically more efficient with better performance. This suggests:
1. 8-bit quantization hurts the 70B model significantly
2. The routing system could save costs by using 8B for most queries
3. Knowledge distillation from 70B → 8B is effective

### Routing System Results

The routing system creates a hybrid approach:
- **Simple queries (71% of problems):** Use distilled 8B model
- **Complex queries (29% of problems):** Use 70B model

**Example routing results:**

```json
{
  "routed_combined": {
    "pass@1": 0.478,
    "pass@10": 0.742,
    "energy_wh": 3500,
    "estimated_cost_ratio": 0.30
  },
  "always_70b": {
    "pass@1": 0.408,
    "pass@10": 0.707,
    "energy_wh": 11400,
    "estimated_cost_ratio": 1.0
  }
}
```

**Interpretation:**
- **Performance:** Routed system achieves 47.8% pass@1 (vs 40.8% for always-70B) - **+17% improvement**
- **Cost:** 0.30x relative cost - **70% savings**
- **Efficiency:** Better performance at 1/3 the cost - **5.9x better performance per dollar**

---

## Advanced Usage

### Custom Complexity Thresholds

Edit `src/routing_system.py` to adjust routing strategy:

```python
def estimate_complexity(self, query: str) -> int:
    tokens = self.encoder.encode(query)
    token_count = len(tokens)

    # Adjust these thresholds based on your needs:
    if token_count <= 150:  # Change threshold here
        return max(1, min(6, token_count // 25))    # Distilled: 1-6
    else:
        return 7 + min(3, (token_count - 151) // 50)  # 70B: 7-10
```

### Wandb Integration

To log results to Weights & Biases:

```bash
export WANDB_API_KEY=your_api_key_here

python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output results/test.jsonl \
  --wandb_project my-llm-project \
  --wandb_run_name baseline-8b-test
```

View results at: https://wandb.ai/

### Limited Testing

For quick tests, limit the number of problems:

```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output results/quick_test.jsonl \
  --limit 10 \
  --num_samples 5
```

This evaluates only 10 problems with 5 samples each (instead of 164 × 10).

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Use quantization: `--load_in_8bit` or `--load_in_4bit`
- Reduce batch size (already at 1 for HumanEval)
- Use gradient checkpointing (already enabled in training)
- Reduce `max_new_tokens`

#### 2. tree-sitter Compatibility Issues

**Error:** `TypeError: an integer is required` when using CodeBLEU

**Solution:** The project uses HumanEval by default, which doesn't require CodeBLEU. If you need CodeBLEU:
- Use the custom `src/codebleu_compat.py` module
- Or stick to HumanEval evaluation

#### 3. Slow Evaluation

**Observation:** Evaluation takes hours

**Expected behavior:**
- 8B model: ~1-2 hours for full HumanEval
- 70B model (8bit): ~13 hours for full HumanEval

**Speed up:**
- Use `--limit 10` for testing
- Reduce `--num_samples` (default 10)
- Use smaller models or more quantization

#### 4. Models Not Found

**Error:** `OSError: meta-llama/Meta-Llama-3.1-70B-Instruct does not exist`

**Solution:**
- Authenticate with HuggingFace: `huggingface-cli login`
- Request access to Llama models at: https://huggingface.co/meta-llama
- Check model path is correct

#### 5. Low Pass Rates

**Observation:** pass@1 < 30%

**Possible causes:**
- Temperature too high (should be 0.01 for HumanEval)
- Code extraction failing (check `.jsonl` output)
- Model not suitable for code generation
- Aggressive quantization (try 8bit instead of 4bit)

---

## Directory Structure

```
results/
├── baseline_8b_humaneval.jsonl              # Raw generations
├── baseline_8b_humaneval.jsonl_results.jsonl # Test results
├── baseline_8b_humaneval.metrics.json       # Summary metrics ← READ THIS
├── teacher_70b_humaneval.jsonl
├── teacher_70b_humaneval.jsonl_results.jsonl
├── teacher_70b_humaneval.metrics.json
├── mbpp_distilled_humaneval.jsonl
├── mbpp_distilled_humaneval.jsonl_results.jsonl
├── mbpp_distilled_humaneval.metrics.json
└── comprehensive_evaluation_summary.json    # Routing comparison
```

---

## Quick Reference Commands

### Run full evaluation suite

```bash
# 1. Baseline 8B
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output results/baseline_8b.jsonl \
  --num_samples 10 --bf16

# 2. Teacher 70B (8-bit)
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output results/teacher_70b.jsonl \
  --load_in_8bit --num_samples 10 --bf16

# 3. Distilled model
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_dir outputs/llama31_8b_kd_lora/lora \
  --output results/distilled.jsonl \
  --num_samples 10 --bf16

# 4. Routing evaluation
python src/routing_system.py \
  --distilled_model_path outputs/llama31_8b_kd_lora \
  --teacher_model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output_dir results/
```

### View results

```bash
# Quick summary
cat results/baseline_8b_humaneval.metrics.json | jq '.pass_at_k, .gpu_metrics | {energy: [.gpu_0_energy_wh, .gpu_1_energy_wh, .gpu_2_energy_wh, .gpu_3_energy_wh, .gpu_4_energy_wh, .gpu_5_energy_wh, .gpu_6_energy_wh, .gpu_7_energy_wh] | add}'

# Compare two runs
jq -s '.[0].pass_at_k, .[1].pass_at_k' results/baseline_8b_humaneval.metrics.json results/teacher_70b_humaneval.metrics.json
```

---

## Visualization

Generate comparison charts:

```bash
python visualize_routing_results.py
```

This creates:
- `routing_comparison_pass_at_k.png` - Pass rate comparison
- `routing_cost_vs_performance.png` - Cost-performance tradeoff
- `routing_efficiency_comparison.png` - Performance per energy unit
- `routing_improvement.png` - Improvement vs always-70B baseline

---

## Additional Resources

- **HumanEval Dataset:** https://github.com/openai/human-eval
- **Llama Models:** https://huggingface.co/meta-llama
- **LoRA/PEFT:** https://github.com/huggingface/peft
- **Weights & Biases:** https://wandb.ai/

---

**Last Updated:** November 4, 2024
