# LLM Compression Pipeline: Complete Step-by-Step Breakdown

## Overview

This document explains **exactly** what happens in the LLM compression pipeline, which files are involved, what technologies are used, and what models are doing what.

---

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Phase 1: Data Preparation](#phase-1-data-preparation)
3. [Phase 2: Knowledge Distillation Training](#phase-2-knowledge-distillation-training)
4. [Phase 3: Evaluation](#phase-3-evaluation)
5. [Phase 4: Routing System](#phase-4-routing-system)
6. [Technologies Used](#technologies-used)

---

## The Big Picture

### What is the Goal?

**Create a cost-efficient code generation system** that combines:
- A **small, fast, cheap model** (8B parameters) for simple tasks
- A **large, powerful, expensive model** (70B parameters) for complex tasks

### The Process (4 Phases)

```
┌─────────────────────┐
│  Phase 1: Data Prep │  Download & format MBPP dataset
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 2: Training  │  Teach 8B model from 70B model (Knowledge Distillation)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Phase 3: Evaluation │  Test all models on HumanEval benchmark
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 4: Routing   │  Combine models intelligently based on query complexity
└─────────────────────┘
```

---

## Phase 1: Data Preparation

### Purpose
Download and format training data for teaching the small model.

### File Involved
- **`src/data_prep.py`** - Main data preparation script

### What Happens

#### Step 1.1: Download MBPP Dataset

```python
from datasets import load_dataset
dataset = load_dataset("mbpp", "full")
```

**MBPP (Mostly Basic Python Problems):**
- Source: HuggingFace Datasets (`mbpp`)
- Contains: ~974 Python programming problems
- Format: Problem description → Solution code + test cases
- Example problem: "Write a python function to print positive numbers in a list"

#### Step 1.2: Convert to Chat Format

**Input (MBPP raw format):**
```json
{
  "text": "Write a python function to print positive numbers in a list.",
  "code": "def pos_nos(list1):\n  for num in list1:\n    if num >= 0:\n      return num",
  "test_list": ["assert pos_nos([1, -2, 3]) == 1"]
}
```

**Output (Chat conversation format):**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful Python coding assistant. Write clean, working code based on the given instructions."
    },
    {
      "role": "user",
      "content": "Write a python function to print positive numbers in a list."
    },
    {
      "role": "assistant",
      "content": "def pos_nos(list1):\n  for num in list1:\n    if num >= 0:\n      return num"
    }
  ],
  "task_id": 313
}
```

#### Step 1.3: Split into Train/Val/Test

**Files created:**
- `data/mbpp_train.jsonl` - 779 examples (80%) - **Used for training**
- `data/mbpp_val.jsonl` - 97 examples (10%) - **Used for validation during training**
- `data/mbpp_test.jsonl` - 98 examples (10%) - **Held out for testing**

**Total:** 974 examples

### Command to Run

```bash
python -m src.data_prep
```

### Technologies Used
- **HuggingFace Datasets** - For downloading MBPP
- **Python JSON** - For data formatting

---

## Phase 2: Knowledge Distillation Training

### Purpose
Train a small 8B model to mimic a large 70B model (make it "learn" from the big model).

### Files Involved
- **`src/train_kd.py`** - Knowledge distillation training script
- **`src/utils.py`** - Helper functions (config loading, chat formatting)
- **`configs/project.yaml`** - Configuration parameters

### The Models

#### Teacher Model (The Expert)
- **Name:** `meta-llama/Meta-Llama-3.1-70B-Instruct`
- **Size:** 70 billion parameters (~140GB in full precision)
- **Role:** Provides "soft targets" (probability distributions) for the student to learn from
- **Quantization:** 8-bit (reduces to ~70GB) for memory efficiency
- **Status:** Frozen (not updated during training)

#### Student Model (The Learner)
- **Name:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Size:** 8 billion parameters (~16GB in full precision)
- **Role:** Learns to match the teacher's behavior
- **Training Method:** LoRA (Low-Rank Adaptation)
- **Status:** Updated during training (only LoRA adapters, not full weights)

### What Happens During Training

#### Step 2.1: Load Models

**Teacher (70B):**
```python
teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    load_in_8bit=True,        # Quantize to 8-bit to save memory
    device_map="auto",         # Spread across multiple GPUs
    torch_dtype=torch.bfloat16
)
teacher.eval()  # Freeze - no training
```

**Student (8B):**
```python
student = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Add LoRA adapters (only train these, not full model)
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling
    target_modules=[         # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ],
    lora_dropout=0.05
)

student = get_peft_model(student, lora_config)
```

**Key Point:** Only the LoRA adapters (~161MB) are trained, not the full 8B model (~16GB). This is **100x more memory efficient**.

#### Step 2.2: Training Loop (Per Example)

**Input:** One training example from `mbpp_train.jsonl`

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful Python coding assistant..."},
    {"role": "user", "content": "Write a python function to print positive numbers in a list."},
    {"role": "assistant", "content": "def pos_nos(list1):\n  for num in list1:\n    if num >= 0:\n      return num"}
  ]
}
```

**Step 2.2.1: Format as Text**
```python
text = tokenizer.apply_chat_template(messages)
```

Result:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful Python coding assistant...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Write a python function to print positive numbers in a list.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
def pos_nos(list1):
  for num in list1:
    if num >= 0:
      return num<|eot_id|>
```

**Step 2.2.2: Tokenize**
```python
tokens = tokenizer.encode(text)  # Convert to token IDs
```

Result: `[1, 2564, 8932, 1234, ...]` (list of integers)

**Step 2.2.3: Forward Pass Through Both Models**

```python
# Teacher produces probability distribution (soft targets)
with torch.no_grad():  # Don't compute gradients
    teacher_logits = teacher(tokens).logits  # Shape: [seq_len, vocab_size]
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

# Student produces its own predictions
student_logits = student(tokens).logits
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
```

**What are logits?** Raw scores for each token in vocabulary (128,256 tokens for Llama).

**Example at one position:**
```
Position: Predicting next token after "def "

Teacher logits:  [0.1, 0.2, 7.5, 0.3, ...]  (128,256 numbers)
                              ↑ (token for "pos_nos")
                          highest score

After softmax → Teacher probs: [0.001, 0.002, 0.95, 0.003, ...]
                                              ↑ 95% confidence

Student logits:  [0.2, 0.3, 5.1, 0.4, ...]
After softmax → Student probs: [0.01, 0.02, 0.80, 0.03, ...]
                                             ↑ 80% confidence

Goal: Make student's 80% → 95% (match teacher)
```

**Step 2.2.4: Compute Knowledge Distillation Loss**

```python
# KL Divergence: How different are student and teacher distributions?
kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

# Cross-Entropy: How well does student predict the actual next token?
ce_loss = F.cross_entropy(student_logits, target_tokens)

# Combined loss (50% each by default)
total_loss = 0.5 * kl_loss * (temperature ** 2) + 0.5 * ce_loss
```

**Why both losses?**
- **KL loss:** Learn from teacher's "soft" knowledge (when teacher is uncertain, learn that too)
- **CE loss:** Stay grounded in the correct answer

**Step 2.2.5: Backpropagation & Update LoRA Weights**

```python
total_loss.backward()  # Compute gradients
optimizer.step()       # Update only LoRA adapter weights
```

#### Step 2.3: Training Configuration

**From `configs/project.yaml`:**
```yaml
training:
  epochs: 10                      # Train for 10 full passes over data
  per_device_batch_size: 1        # 1 example at a time per GPU
  grad_accum: 16                  # Accumulate 16 examples before updating
  learning_rate: 2.0e-4           # How fast to update weights
  kd_alpha: 0.5                   # 50% KD loss, 50% CE loss
  kd_temp: 1.0                    # Temperature for softening distributions
```

**Effective batch size:** 1 × 16 (grad accum) × 8 (GPUs) = 128 examples per update

**Training time:** ~6-12 hours on 8× A100 GPUs

#### Step 2.4: Validation During Training

After each epoch, evaluate on `mbpp_val.jsonl`:

**Metrics tracked:**
- **Validation loss** - How well student predicts validation examples
- **Token accuracy** - % of tokens where student's top prediction matches teacher's
- **Top-5 agreement** - % of tokens where student's top-5 contains teacher's top choice
- **KL divergence** - Average distribution difference

**Example output:**
```
Epoch 1/10: val_loss=1.234, token_acc=68.5%, top5_agree=91.2%, kl_div=0.089
Epoch 5/10: val_loss=0.891, token_acc=75.3%, top5_agree=94.1%, kl_div=0.065
Epoch 10/10: val_loss=0.823, token_acc=78.2%, top5_agree=95.8%, kl_div=0.058
```

#### Step 2.5: Save LoRA Adapters

**Output directory:** `outputs/llama31_8b_kd_lora/lora/`

**Files created:**
- **`adapter_model.safetensors`** (161MB) - The trained LoRA weights
- **`adapter_config.json`** - LoRA configuration
- **`tokenizer.json`** (17MB) - Tokenizer for the model
- **`special_tokens_map.json`** - Special tokens configuration

**Total size:** ~177MB (vs 16GB for full model!)

### Command to Run

```bash
python -m src.train_kd \
  --teacher_model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --student_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train_file data/mbpp_train.jsonl \
  --val_file data/mbpp_val.jsonl \
  --output_dir outputs/llama31_8b_kd_lora \
  --num_epochs 10 \
  --bf16 True \
  --teacher_8bit True
```

### Technologies Used
- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - Model loading and training
- **PEFT (Parameter-Efficient Fine-Tuning)** - LoRA implementation
- **BitsAndBytes** - 8-bit quantization
- **Knowledge Distillation** - Training technique (KL divergence)

---

## Phase 3: Evaluation

### Purpose
Measure how well models solve real programming problems.

### File Involved
- **`src/eval_humaneval.py`** - HumanEval evaluation script

### The Benchmark: HumanEval

**What is HumanEval?**
- Created by OpenAI
- Contains: 164 hand-written Python programming problems
- Difficulty: Entry-level to intermediate software engineering
- Each problem has: Function signature, docstring, test cases

**Example HumanEval problem:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Task:** Generate the function body that passes all test cases.

### What Happens During Evaluation

#### Step 3.1: Load Model

**Option A: Base 8B Model**
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

**Option B: Distilled Model (8B + LoRA)**
```python
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters on top
from peft import PeftModel
model = PeftModel.from_pretrained(
    model,
    "outputs/llama31_8b_kd_lora/lora"
)
```

**Option C: 70B Model (8-bit quantized)**
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

#### Step 3.2: For Each Problem (164 total)

**Step 3.2.1: Load Problem**
```python
from human_eval.data import read_problems

problems = read_problems()  # Returns dict of 164 problems
problem = problems["HumanEval/0"]
```

**Example:**
```python
{
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers...\n    \"\"\"",
    "entry_point": "has_close_elements",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    ..."
}
```

**Step 3.2.2: Format as Chat**
```python
messages = [
    {"role": "system", "content": "You are a Python coding assistant."},
    {"role": "user", "content": problem["prompt"]}
]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

**Step 3.2.3: Generate 10 Solutions (for pass@k metrics)**

```python
for i in range(10):  # Generate 10 different solutions
    output = model.generate(
        input_ids=tokenizer.encode(prompt_text),
        max_new_tokens=512,
        temperature=0.01,  # Nearly deterministic (greedy)
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    completion = tokenizer.decode(output[0])
    solutions.append(completion)
```

**Why 10 solutions?** To compute pass@k metrics:
- **pass@1:** Does the first solution work?
- **pass@5:** Does at least one of the first 5 work?
- **pass@10:** Does at least one of all 10 work?

**Step 3.2.4: Extract Code**

The model generates more than just code:

**Raw generation:**
```
<|start_header_id|>assistant<|end_header_id|>

Here's the solution:

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

This function checks all pairs of numbers...<|eot_id|>
```

**Extracted code (via `extract_code_from_completion`):**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Extraction logic:**
1. Remove markdown code fences (` ```python `)
2. Remove "assistant" markers
3. Find the function definition (`def `)
4. Remove docstrings
5. Keep only function body

**Step 3.2.5: Test Solutions**

```python
from human_eval.evaluation import evaluate_functional_correctness

# Create JSONL file with all generations
write_jsonl("results/baseline_8b_humaneval.jsonl", [
    {
        "task_id": "HumanEval/0",
        "completion": extracted_code_1
    },
    {
        "task_id": "HumanEval/0",
        "completion": extracted_code_2
    },
    # ... 10 solutions for HumanEval/0
    # ... then 10 solutions for HumanEval/1
    # ... etc for all 164 problems
])

# Run evaluation (executes code and checks tests)
results = evaluate_functional_correctness("results/baseline_8b_humaneval.jsonl")
```

**What happens in testing:**
1. For each solution, execute it in a sandboxed environment
2. Run the test cases (e.g., `assert candidate([1.0, 2.0, 3.0], 0.5) == False`)
3. Mark as **passed** if all tests pass, **failed** otherwise
4. Record results

**Example results file (`baseline_8b_humaneval.jsonl_results.jsonl`):**
```jsonl
{"task_id": "HumanEval/0", "completion": "def has_close_elements(...):\n    ...", "passed": true, "result": "passed"}
{"task_id": "HumanEval/0", "completion": "def has_close_elements(...):\n    ...", "passed": false, "result": "failed: AssertionError"}
{"task_id": "HumanEval/0", "completion": "def has_close_elements(...):\n    ...", "passed": true, "result": "passed"}
...
```

#### Step 3.3: GPU Monitoring (Parallel Thread)

**While generating solutions, track GPU metrics every 100ms:**

```python
import pynvml
pynvml.nvmlInit()

while generating:
    for gpu_id in range(8):  # 8 GPUs
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        # Power consumption
        power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

        # GPU utilization %
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util_percent = util.gpu

        # Memory usage
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_gb = mem_info.used / (1024**3)

        # Temperature
        temp_c = pynvml.nvmlDeviceGetTemperature(handle, 0)

        metrics.append({
            'timestamp': time.time(),
            'gpu_id': gpu_id,
            'power_watts': power_watts,
            'utilization': gpu_util_percent,
            'memory_gb': memory_used_gb,
            'temperature': temp_c
        })

    time.sleep(0.1)  # 100ms interval
```

#### Step 3.4: Compute Metrics

**Performance metrics:**
```python
# pass@k calculation
def pass_at_k(n_samples, n_correct, k):
    """
    Probability that at least one of k samples is correct.
    """
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n_samples - n_correct + 1, n_samples + 1))

# For each problem, count how many of 10 solutions passed
results_by_problem = group_by_task_id(results)

pass_at_1_scores = []
pass_at_5_scores = []
pass_at_10_scores = []

for task_id, task_results in results_by_problem.items():
    n_samples = 10
    n_correct = sum(1 for r in task_results if r['passed'])

    pass_at_1_scores.append(pass_at_k(n_samples, n_correct, 1))
    pass_at_5_scores.append(pass_at_k(n_samples, n_correct, 5))
    pass_at_10_scores.append(pass_at_k(n_samples, n_correct, 10))

pass_at_1 = np.mean(pass_at_1_scores)   # Average across 164 problems
pass_at_5 = np.mean(pass_at_5_scores)
pass_at_10 = np.mean(pass_at_10_scores)
```

**Energy metrics:**
```python
total_time_hours = (end_time - start_time) / 3600

for gpu_id in range(8):
    gpu_power_measurements = [m['power_watts'] for m in metrics if m['gpu_id'] == gpu_id]
    avg_power = np.mean(gpu_power_measurements)

    # Energy = Power × Time
    energy_wh = avg_power * total_time_hours

    gpu_metrics[f'gpu_{gpu_id}_energy_wh'] = energy_wh
```

#### Step 3.5: Save Results

**Three files created:**

**1. Raw generations:** `baseline_8b_humaneval.jsonl`
```jsonl
{"task_id": "HumanEval/0", "prompt": "def has_close_elements...", "completion": "    for i in range..."}
{"task_id": "HumanEval/0", "prompt": "def has_close_elements...", "completion": "    for idx, elem..."}
...
```

**2. Test results:** `baseline_8b_humaneval.jsonl_results.jsonl`
```jsonl
{"task_id": "HumanEval/0", "passed": true, "result": "passed"}
{"task_id": "HumanEval/0", "passed": false, "result": "failed: AssertionError"}
...
```

**3. Summary metrics:** `baseline_8b_humaneval.metrics.json`
```json
{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "num_problems": 164,
  "num_samples_per_problem": 10,
  "total_generations": 1640,
  "pass_at_k": {
    "pass@1": 0.4951,
    "pass@5": 0.7199,
    "pass@10": 0.7561
  },
  "gpu_metrics": {
    "gpu_0_energy_wh": 88.99,
    "gpu_1_energy_wh": 93.84,
    ...
  },
  "generation_time_sec": 3729.35,
  "avg_time_per_sample": 2.27
}
```

### Command to Run

**Baseline 8B:**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output results/baseline_8b_humaneval.jsonl \
  --num_samples 10 \
  --bf16
```

**Distilled 8B:**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_dir outputs/llama31_8b_kd_lora/lora \
  --output results/distilled_humaneval.jsonl \
  --num_samples 10 \
  --bf16
```

**70B (8-bit):**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output results/teacher_70b_humaneval.jsonl \
  --load_in_8bit \
  --num_samples 10 \
  --bf16
```

### Technologies Used
- **HumanEval** - Benchmark dataset
- **PyTorch** - Model inference
- **HuggingFace Transformers** - Model generation
- **pynvml** - GPU monitoring (NVIDIA Management Library)
- **Code execution sandbox** - Safe test execution

---

## Phase 4: Routing System

### Purpose
Combine small and large models intelligently to maximize performance while minimizing cost.

### File Involved
- **`src/routing_system.py`** - Routing evaluation script

### The Strategy

**Hypothesis:** Simple problems don't need a 70B model. Route them to 8B to save cost.

**Routing decision:**
1. Estimate problem complexity (1-10 scale)
2. If complexity ≤ 6: Use distilled 8B model
3. If complexity ≥ 7: Use 70B model

### What Happens

#### Step 4.1: Classify All HumanEval Problems by Complexity

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

for problem in humaneval_problems:
    prompt = problem["prompt"]

    # Count tokens in the prompt
    tokens = encoder.encode(prompt)
    token_count = len(tokens)

    # Map token count to complexity (1-10)
    if token_count <= 150:
        complexity = max(1, min(6, token_count // 25))
    else:
        complexity = 7 + min(3, (token_count - 151) // 50)

    problem["complexity"] = complexity
```

**Example:**
```python
# Problem 1: "Write a function to sum two numbers"
# Token count: 25 → Complexity: 1 (Simple) → Route to 8B

# Problem 2: "Implement a function to find the longest palindromic substring..."
# Token count: 180 → Complexity: 7 (Complex) → Route to 70B
```

**Distribution:**
- Complexity 1-6 (Simple): ~117 problems (71%)
- Complexity 7-10 (Complex): ~47 problems (29%)

#### Step 4.2: Run 4 Separate Evaluations

**1. Distilled 8B on Simple (1-6):**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_dir outputs/llama31_8b_kd_lora/lora \
  --output results/distilled_simple.jsonl \
  --problem_filter complexity_1_6
```

**2. Distilled 8B on Complex (7-10):**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --lora_dir outputs/llama31_8b_kd_lora/lora \
  --output results/distilled_complex.jsonl \
  --problem_filter complexity_7_10
```

**3. 70B on Simple (1-6):**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --load_in_8bit \
  --output results/70b_simple.jsonl \
  --problem_filter complexity_1_6
```

**4. 70B on Complex (7-10):**
```bash
python src/eval_humaneval.py \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --load_in_8bit \
  --output results/70b_complex.jsonl \
  --problem_filter complexity_7_10
```

#### Step 4.3: Create Routed Results

**Combine the best of both:**
```python
routed_results = []

for problem in humaneval_problems:
    if problem["complexity"] <= 6:
        # Use distilled 8B result
        result = distilled_simple_results[problem["task_id"]]
    else:
        # Use 70B result
        result = seventyb_complex_results[problem["task_id"]]

    routed_results.append(result)
```

#### Step 4.4: Compute Combined Metrics

**Performance:**
```python
routed_pass_at_1 = compute_pass_at_k(routed_results, k=1)
routed_pass_at_10 = compute_pass_at_k(routed_results, k=10)
```

**Cost:**
```python
# 71% of problems use 8B, 29% use 70B
routed_energy = (
    0.71 * distilled_simple_energy +
    0.29 * seventyb_complex_energy
)

# Compare to always using 70B
always_70b_energy = (
    seventyb_simple_energy +
    seventyb_complex_energy
)

savings = 1 - (routed_energy / always_70b_energy)
print(f"Energy savings: {savings * 100:.1f}%")
```

#### Step 4.5: Generate Visualizations

**Charts created (via `visualize_routing_results.py`):**

1. **Pass@k comparison:** Bar chart
   - Baseline 8B vs Distilled 8B vs 70B vs Routed

2. **Cost vs Performance:** Scatter plot
   - X-axis: Energy consumption (Wh)
   - Y-axis: Pass@1 accuracy
   - Point size: Model size

3. **Efficiency:** Horizontal bars
   - Performance per Watt-hour

4. **Routing improvement:** % better than always-70B

### Command to Run

```bash
python src/routing_system.py \
  --distilled_model_path outputs/llama31_8b_kd_lora \
  --teacher_model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --output_dir results/ \
  --wandb_project llm-routing-eval
```

### Technologies Used
- **tiktoken** - Token counting (OpenAI's tokenizer)
- **matplotlib/seaborn** - Visualization
- **pandas** - Data manipulation

---

## Technologies Used (Summary)

| Technology | Purpose | Used In |
|------------|---------|---------|
| **PyTorch** | Deep learning framework | Training, Evaluation |
| **HuggingFace Transformers** | Model loading, tokenization, generation | All phases |
| **HuggingFace Datasets** | Dataset downloading | Data prep |
| **PEFT (LoRA)** | Parameter-efficient fine-tuning | Training |
| **BitsAndBytes** | 8-bit/4-bit quantization | Training, Evaluation |
| **Knowledge Distillation (KL divergence)** | Training technique | Training |
| **HumanEval** | Code generation benchmark | Evaluation |
| **pynvml** | GPU monitoring | Evaluation |
| **tiktoken** | Token counting for routing | Routing |
| **matplotlib/seaborn** | Visualization | Routing |
| **Weights & Biases (wandb)** | Experiment tracking (optional) | Training, Evaluation |

---

## Models Summary

| Model | Size | Parameters | Purpose | Memory (bf16) | Memory (8-bit) |
|-------|------|------------|---------|---------------|----------------|
| **Meta-Llama-3.1-70B-Instruct** | 70B | 70 billion | Teacher model (provides knowledge) | ~140GB | ~70GB |
| **Meta-Llama-3.1-8B-Instruct** | 8B | 8 billion | Student model (learns from teacher) | ~16GB | ~8GB |
| **Distilled 8B (with LoRA)** | 8B + adapters | 8B + 84M | Trained student (production model) | ~16GB + 161MB | N/A |

**Key insight:** The LoRA adapters are only **161MB**, making the distilled model extremely portable!

---

## File Flow Diagram

```
Data Preparation:
  HuggingFace → src/data_prep.py → data/mbpp_train.jsonl (779 examples)
                                   → data/mbpp_val.jsonl (97 examples)
                                   → data/mbpp_test.jsonl (98 examples)

Training:
  data/mbpp_train.jsonl + configs/project.yaml → src/train_kd.py → outputs/llama31_8b_kd_lora/lora/
                                                                      ├── adapter_model.safetensors (161MB)
                                                                      ├── adapter_config.json
                                                                      └── tokenizer files

Evaluation:
  HumanEval (164 problems) + Model → src/eval_humaneval.py → results/
                                                               ├── <name>.jsonl (raw generations)
                                                               ├── <name>.jsonl_results.jsonl (test results)
                                                               └── <name>.metrics.json (summary)

Routing:
  All evaluation results → src/routing_system.py → results/comprehensive_evaluation_summary.json
                                                   → visualize_routing_results.py → charts/*.png
```

---

## Quick Reference: What Each Model Does

### 1. **Teacher (70B) - The Expert**
- **During Training:** Generates "soft targets" (probability distributions) for student to learn from
- **During Evaluation:** Solves HumanEval problems directly (baseline for comparison)
- **Status:** Pre-trained, not modified

### 2. **Student (8B) - The Learner**
- **During Training:** Learns to mimic teacher's behavior using LoRA adapters
- **During Evaluation:** Can be tested as:
  - Baseline (no training)
  - With LoRA adapters (distilled version)
- **Status:** Base model pre-trained, LoRA adapters trained by you

### 3. **Distilled (8B + LoRA) - The Production Model**
- **During Training:** Created from student + teacher
- **During Evaluation:** Tested on HumanEval to measure effectiveness
- **During Routing:** Used for 71% of problems (simple ones)
- **Status:** Your final compressed model

---

**Last Updated:** November 4, 2024
