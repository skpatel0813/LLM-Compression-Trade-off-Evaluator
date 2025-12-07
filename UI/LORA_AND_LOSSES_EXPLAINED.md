# LoRA Adapters vs Distillation Loss, Ground Truth Loss, and Total Loss

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [What are LoRA Adapters?](#what-are-lora-adapters)
3. [The Three Losses Explained](#the-three-losses-explained)
4. [How They Work Together](#how-they-work-together)
5. [Mathematical Deep Dive](#mathematical-deep-dive)
6. [Visual Example](#visual-example)

---

## The Big Picture

### Two Different Concepts Working Together

**LoRA Adapters** and **Loss Functions** serve completely different purposes:

| Aspect | LoRA Adapters | Loss Functions |
|--------|---------------|----------------|
| **What?** | A **training technique** (HOW we update the model) | **Objectives** (WHAT we're trying to achieve) |
| **Purpose** | Make training memory-efficient | Guide the model to learn the right thing |
| **Question answered** | "Which weights should we update?" | "How well is the model doing?" |
| **Type** | Architectural modification | Training signal |

**Analogy:**
- **LoRA** = The **tool** you use (e.g., a small paintbrush instead of repainting the whole wall)
- **Losses** = The **goal** you're trying to achieve (e.g., match the teacher's painting style)

---

## What are LoRA Adapters?

### The Problem: Full Fine-Tuning is Expensive

**Traditional fine-tuning:**
```
Meta-Llama-3.1-8B has 8 billion parameters
Each parameter = 2 bytes (bf16 precision)
Total model size = 16 GB

During training, you need:
- Model weights: 16 GB
- Gradients: 16 GB (same size as weights)
- Optimizer states (Adam): 32 GB (2x weights)
Total = 64 GB just for one 8B model!
```

**Problem:** Can't fit on most GPUs, very slow to train, hard to store/share.

### The Solution: LoRA (Low-Rank Adaptation)

**Key insight:** You don't need to update ALL 8 billion parameters. Most changes happen in a low-dimensional subspace.

#### How LoRA Works

**Instead of updating the full weight matrix, add small "adapter" matrices.**

**Original weight matrix (frozen):**
```
W_original ∈ ℝ^(4096 × 4096)  (16.7 million parameters)
```

**LoRA adds two small matrices:**
```
W_lora = W_original + B × A

Where:
  A ∈ ℝ^(4096 × 16)   (65,536 parameters)
  B ∈ ℝ^(16 × 4096)   (65,536 parameters)

Total LoRA params: 131,072 (vs 16.7 million original)
Reduction: 128× fewer parameters!
```

**Visual representation:**
```
Original matrix (4096 × 4096):
┌─────────────────────────┐
│ ████████████████████████│
│ ████████████████████████│  16.7M parameters
│ ████████████████████████│  (FROZEN - not updated)
│ ████████████████████████│
└─────────────────────────┘

LoRA decomposition:
┌─────────────┐   ┌─────────────────────────┐
│ ████████████│   │ ████████████████████████│
│ ████████████│ × │                         │  131K parameters
│ ████████████│   │                         │  (TRAINABLE)
│ ████████████│   │                         │
└─────────────┘   └─────────────────────────┘
   Matrix B              Matrix A
  (16 × 4096)         (4096 × 16)
```

#### LoRA in Your Project

**Configuration (from `src/train_kd.py`):**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank of decomposition (size of bottleneck)
    lora_alpha=32,           # Scaling factor (α/r = 2.0)
    target_modules=[         # Which layers to add adapters to
        # Attention layers
        "q_proj",   # Query projection
        "k_proj",   # Key projection
        "v_proj",   # Value projection
        "o_proj",   # Output projection

        # MLP layers
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,       # Dropout for regularization
    task_type="CAUSAL_LM"
)

# Apply LoRA to the student model
student_model = get_peft_model(base_model, lora_config)
```

**What gets trained:**
```
Base Llama-3.1-8B: 8,000,000,000 parameters (FROZEN ❄️)
LoRA adapters:        84,000,000 parameters (TRAINABLE 🔥)

Trainable percentage: 1.05% of total parameters
```

**Memory savings:**
```
Full fine-tuning:
  - Trainable params: 8B
  - Memory required: ~64 GB

LoRA fine-tuning:
  - Trainable params: 84M
  - Memory required: ~18 GB (base model + small adapters + gradients)

Reduction: 3.5× less memory!
```

#### What LoRA Does NOT Do

❌ LoRA does NOT determine what the model learns (that's the job of the loss function)
❌ LoRA does NOT decide how to combine teacher and ground truth (that's the loss)
❌ LoRA does NOT affect the training objective

✅ LoRA ONLY determines which parameters get updated during training
✅ LoRA makes training memory-efficient
✅ LoRA produces a small, portable adapter file (161 MB)

---

## The Three Losses Explained

### Overview

During knowledge distillation, we have **three loss components**:

1. **Distillation Loss (KL Divergence)** - Learn from teacher's soft predictions
2. **Ground Truth Loss (Cross-Entropy)** - Learn from the correct answer
3. **Total Loss** - Weighted combination of both

These losses tell the model **WHAT to learn**, while LoRA determines **HOW to update weights**.

---

### 1. Distillation Loss (KL Divergence)

**Purpose:** Make the student's output distribution match the teacher's distribution.

**What it measures:** "How different are the student's predictions from the teacher's?"

#### The Setup

**Vocabulary:** Both models predict over the same vocabulary (128,256 tokens for Llama).

**Example scenario:**
```python
Input sequence: "Write a function to calculate the sum of"
Next token to predict: ?

Teacher thinks:
  - "all" → 45% probability
  - "two" → 30% probability
  - "numbers" → 15% probability
  - "elements" → 8% probability
  - ... (other 128,252 tokens) → 2% total

Student thinks (before training):
  - "the" → 60% probability
  - "all" → 20% probability
  - "two" → 10% probability
  - "numbers" → 5% probability
  - ... (other 128,252 tokens) → 5% total
```

**Problem:** Student's distribution is different from teacher's.

#### The Math

**KL Divergence formula:**
```
KL(P_teacher || P_student) = Σ P_teacher(i) × log(P_teacher(i) / P_student(i))
                             i∈vocab

Where:
  P_teacher = Teacher's probability distribution (soft targets)
  P_student = Student's probability distribution
```

**In code (from `src/train_kd.py`):**
```python
# Get teacher's predictions (with temperature)
with torch.no_grad():  # Don't compute gradients for teacher
    teacher_logits = teacher_model(input_ids)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

# Get student's predictions (with temperature)
student_logits = student_model(input_ids)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

# Compute KL divergence
kl_loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction='batchmean'
)

# Scale by temperature squared (standard practice)
distillation_loss = kl_loss * (temperature ** 2)
```

#### Why Temperature?

**Temperature** (T) softens the probability distributions:

**Without temperature (T=1):**
```
Teacher logits: [0.5, 3.0, 0.2, 0.1]
After softmax:  [0.08, 0.87, 0.06, 0.05]  ← Very peaked (87% on one token)
```

**With temperature (T=2):**
```
Teacher logits / T: [0.25, 1.5, 0.1, 0.05]
After softmax:      [0.15, 0.58, 0.13, 0.14]  ← More spread out
```

**Why this matters:**
- Hard targets (T=1): "The answer is definitely 'all'"
- Soft targets (T=2): "The answer is probably 'all', but 'two' and 'numbers' are also reasonable"

**The soft targets contain more information** about what the teacher "almost" predicted, which helps the student learn better.

#### What Distillation Loss Teaches

✅ Teacher's "dark knowledge" (uncertainty, reasoning patterns)
✅ Which alternatives are reasonable (not just the top choice)
✅ Teacher's biases and preferences
✅ Smooth decision boundaries

---

### 2. Ground Truth Loss (Cross-Entropy)

**Purpose:** Make sure the student predicts the actual correct token.

**What it measures:** "How well does the student predict the real answer?"

#### The Setup

**Ground truth:** The actual next token in the training data.

**Example:**
```python
Input:  "Write a function to calculate the sum of"
Target: "all"  (the actual next token in the training data)
```

**Student's prediction:**
```
Student probabilities:
  - "all" → 45%
  - "two" → 30%
  - "the" → 15%
  - ...
```

**Question:** How confident is the student in the correct answer ("all")?

#### The Math

**Cross-Entropy formula:**
```
CE(y_true, y_pred) = -log(P_student(y_true))

Where:
  y_true = The correct token ID
  P_student(y_true) = Student's probability for the correct token
```

**Interpretation:**
```
If P_student("all") = 0.45:
  CE = -log(0.45) = 0.80

If P_student("all") = 0.90:
  CE = -log(0.90) = 0.11  ← Lower is better!

If P_student("all") = 0.01:
  CE = -log(0.01) = 4.61  ← Very high penalty!
```

**In code:**
```python
# Get student's predictions (NO temperature for ground truth)
student_logits = student_model(input_ids)

# Target = the actual next tokens
targets = input_ids[:, 1:]  # Shift by one position

# Compute cross-entropy loss
ce_loss = F.cross_entropy(
    student_logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=pad_token_id
)

ground_truth_loss = ce_loss
```

#### What Ground Truth Loss Teaches

✅ The factually correct answer
✅ Prevents the student from "drifting" too far from reality
✅ Ensures the model actually solves the task
✅ Anchors learning to the dataset

---

### 3. Total Loss (Weighted Combination)

**Purpose:** Balance learning from the teacher vs learning the ground truth.

#### The Formula

```python
total_loss = alpha × distillation_loss + (1 - alpha) × ground_truth_loss

Where:
  alpha = 0.5 (in your project)

  So:
  total_loss = 0.5 × distillation_loss + 0.5 × ground_truth_loss
```

#### Why Combine Both?

**If we only used distillation loss (alpha=1.0):**
- ❌ Student might copy teacher's mistakes
- ❌ No guarantee of correct answers
- ❌ Could drift from the actual task

**If we only used ground truth loss (alpha=0.0):**
- ❌ Student doesn't learn teacher's reasoning
- ❌ Misses "dark knowledge"
- ❌ Just standard fine-tuning (no distillation)

**With both (alpha=0.5):**
- ✅ Learn teacher's knowledge AND the correct answers
- ✅ Best of both worlds
- ✅ Robust training

#### Configuration

**From `configs/project.yaml`:**
```yaml
training:
  kd_alpha: 0.5      # 50% distillation, 50% ground truth
  kd_temp: 1.0       # Temperature for soft targets
```

**You can adjust these:**
- `kd_alpha=0.7` → More emphasis on teacher (70% distillation, 30% ground truth)
- `kd_alpha=0.3` → More emphasis on correctness (30% distillation, 70% ground truth)
- `kd_temp=2.0` → Softer targets (more information from teacher)

---

## How They Work Together

### The Training Loop (Putting It All Together)

**Step-by-step for one training example:**

```
Input text: "Write a function to sum two numbers"
Target text: "def add(a, b):\n    return a + b"
```

#### Step 1: Forward Pass Through Teacher (Frozen)

```python
with torch.no_grad():  # Don't compute gradients
    teacher_logits = teacher_model(input_tokens)
    # Shape: [sequence_length, vocab_size]
    #        [50, 128256] for this example

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
```

**What happens:** Teacher produces probability distribution for each token position.

**Example at position 10 (predicting "def"):**
```
Teacher probabilities (out of 128,256 tokens):
  token_id_3881 ("def"):   0.78
  token_id_629 ("function"): 0.10
  token_id_8117 ("class"):   0.05
  ... (128,253 other tokens): 0.07
```

#### Step 2: Forward Pass Through Student (Trainable LoRA)

```python
# Student has base model (frozen) + LoRA adapters (trainable)
student_logits = student_model(input_tokens)
# Internally: logits = frozen_base(x) + lora_B @ lora_A @ x

student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
student_probs_ce = F.softmax(student_logits, dim=-1)  # No temp for CE
```

**What happens:** Student (with current LoRA weights) produces its own predictions.

**Example at position 10:**
```
Student probabilities (BEFORE training step):
  token_id_3881 ("def"):   0.45
  token_id_629 ("function"): 0.30
  token_id_8117 ("class"):   0.15
  ... (128,253 other tokens): 0.10
```

#### Step 3: Compute Losses

**3a. Distillation Loss:**
```python
kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
distillation_loss = kl_loss * (temperature ** 2)

# Example value: 0.145
# Interpretation: Student's distribution differs from teacher's
```

**3b. Ground Truth Loss:**
```python
ce_loss = F.cross_entropy(student_logits, target_tokens)
ground_truth_loss = ce_loss

# Example value: 0.802
# Interpretation: -log(0.45) ≈ 0.802
# Student only gives 45% confidence to correct token "def"
```

**3c. Total Loss:**
```python
total_loss = 0.5 * distillation_loss + 0.5 * ground_truth_loss
           = 0.5 * 0.145 + 0.5 * 0.802
           = 0.0725 + 0.401
           = 0.474
```

#### Step 4: Backpropagation (Update LoRA Adapters)

```python
total_loss.backward()  # Compute gradients

# Gradients computed for:
# ✅ LoRA matrices A and B (trainable)
# ❌ Base model weights (frozen)

optimizer.step()  # Update only LoRA parameters
```

**What changes:**
```
Before step:
  lora_A[0, 0] = 0.0234
  lora_B[0, 0] = -0.0156

Gradient:
  grad_A[0, 0] = -0.0012  (loss wants to increase this)
  grad_B[0, 0] = 0.0008   (loss wants to decrease this)

After step (learning_rate = 0.0002):
  lora_A[0, 0] = 0.0234 + 0.0002 * (-0.0012) = 0.02337
  lora_B[0, 0] = -0.0156 + 0.0002 * (0.0008) = -0.01561
```

#### Step 5: Next Forward Pass (Improved Predictions)

**After update, student's new predictions:**
```
Student probabilities (AFTER training step):
  token_id_3881 ("def"):   0.52  ← Improved from 0.45
  token_id_629 ("function"): 0.25  ← Reduced from 0.30
  token_id_8117 ("class"):   0.13  ← Reduced from 0.15
  ... (other tokens): 0.10
```

**Why the improvement?**
1. **Distillation loss** pushed student toward teacher's 0.78
2. **Ground truth loss** pushed student to increase probability of "def"
3. **LoRA adapters** were updated to make this happen

---

## Mathematical Deep Dive

### The Complete Forward Pass with LoRA

**Original transformer layer (frozen):**
```
h = input_hidden_states  # Shape: [batch, seq_len, 4096]

# Self-attention
Q = h @ W_q  # W_q is frozen
K = h @ W_k  # W_k is frozen
V = h @ W_v  # W_v is frozen
attention_output = Attention(Q, K, V) @ W_o  # W_o is frozen

# MLP
mlp_output = GELU(h @ W_gate) * (h @ W_up) @ W_down  # All frozen

output = attention_output + mlp_output
```

**With LoRA (trainable):**
```
h = input_hidden_states

# Self-attention with LoRA
Q = h @ W_q + h @ A_q @ B_q  # Added LoRA adapters
K = h @ W_k + h @ A_k @ B_k
V = h @ W_v + h @ A_v @ B_v
attention_output = Attention(Q, K, V) @ (W_o + A_o @ B_o)

# MLP with LoRA
gate = h @ W_gate + h @ A_gate @ B_gate
up = h @ W_up + h @ A_up @ B_up
down = W_down + A_down @ B_down
mlp_output = GELU(gate) * up @ down

output = attention_output + mlp_output
```

**Key insight:** The LoRA terms (A @ B) are small corrections to the frozen weights.

### The Gradient Flow

**When we compute `total_loss.backward()`, gradients flow through:**

```
total_loss
  ├── 0.5 × distillation_loss (KL divergence)
  │     └── student_log_probs - teacher_probs
  │           └── student_logits
  │                 └── student_model output
  │                       └── LoRA adapters (A, B matrices) ✅ UPDATED
  │                       └── Base weights ❌ FROZEN
  │
  └── 0.5 × ground_truth_loss (Cross-entropy)
        └── student_logits - target_tokens
              └── student_model output
                    └── LoRA adapters (A, B matrices) ✅ UPDATED
                    └── Base weights ❌ FROZEN
```

### Parameter Count Breakdown

**Your project (Llama-3.1-8B with LoRA r=16):**

```python
Base model parameters (frozen):
  - Embedding layer: 1.05B
  - 32 Transformer layers: 6.74B
  - Output layer: 0.21B
  Total frozen: 8.0B parameters

LoRA adapters (trainable):
  Per layer:
    - q_proj: 4096×16 + 16×4096 = 131,072 params
    - k_proj: 4096×16 + 16×4096 = 131,072 params
    - v_proj: 4096×16 + 16×4096 = 131,072 params
    - o_proj: 4096×16 + 16×4096 = 131,072 params
    - gate_proj: 4096×16 + 16×14336 = 294,912 params
    - up_proj: 4096×16 + 16×14336 = 294,912 params
    - down_proj: 14336×16 + 16×4096 = 294,912 params

  Per layer total: 1,409,024 params
  32 layers: 45,088,768 params

Total trainable: ~45M parameters (0.56% of base model)
```

**Actual file sizes:**
```
Base model:          16 GB (8B params × 2 bytes)
LoRA adapters:       161 MB (84M params × 2 bytes)
Compression ratio:   ~100×
```

---

## Visual Example

### Training Timeline (One Batch)

```
Position in sequence: Predicting next token after "def "
Ground truth token: "add"

┌─────────────────────────────────────────────────────────────┐
│ Before Training Step                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Teacher predictions:                                        │
│   "add"  ████████████████████ 70%                         │
│   "sum"  ██████ 20%                                        │
│   "func" ██ 8%                                             │
│   other  █ 2%                                              │
│                                                             │
│ Student predictions:                                        │
│   "add"  ████████████ 40%  ← Too low!                     │
│   "sum"  ████████████ 40%  ← Too high!                    │
│   "func" ████ 15%          ← Too high!                    │
│   other  ██ 5%                                             │
│                                                             │
│ Distillation loss: 0.215 (student ≠ teacher)              │
│ Ground truth loss: 0.916 (only 40% confidence in "add")   │
│ Total loss:        0.566                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
                  Backpropagation
                            ↓
              Update LoRA adapters A, B
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ After Training Step                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Teacher predictions: (unchanged - frozen)                   │
│   "add"  ████████████████████ 70%                         │
│   "sum"  ██████ 20%                                        │
│   "func" ██ 8%                                             │
│   other  █ 2%                                              │
│                                                             │
│ Student predictions: (improved via LoRA)                    │
│   "add"  ████████████████ 55%  ← Increased! ✅            │
│   "sum"  ██████████ 30%         ← Decreased               │
│   "func" ████ 12%               ← Decreased               │
│   other  █ 3%                                              │
│                                                             │
│ Distillation loss: 0.089 (closer to teacher) ✅            │
│ Ground truth loss: 0.598 (higher confidence) ✅            │
│ Total loss:        0.344 (improved!) ✅                    │
└─────────────────────────────────────────────────────────────┘
```

### After 10 Epochs (Full Training)

```
┌─────────────────────────────────────────────────────────────┐
│ Fully Trained Student                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Teacher predictions:                                        │
│   "add"  ████████████████████ 70%                         │
│   "sum"  ██████ 20%                                        │
│   "func" ██ 8%                                             │
│   other  █ 2%                                              │
│                                                             │
│ Student predictions: (well-aligned)                         │
│   "add"  ██████████████████ 68%  ← Very close! ✅         │
│   "sum"  ██████ 22%              ← Close! ✅              │
│   "func" ██ 8%                   ← Match! ✅              │
│   other  █ 2%                    ← Match! ✅              │
│                                                             │
│ Distillation loss: 0.012 (nearly identical) ✅             │
│ Ground truth loss: 0.385 (68% confidence) ✅               │
│ Total loss:        0.199 (very low!) ✅                    │
│                                                             │
│ Result: Student successfully learned from teacher!          │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: The Relationship

### LoRA Adapters (The MECHANISM)

**Role:** HOW we update the model
- Adds small trainable matrices to frozen base model
- Only 1% of parameters are trainable
- Saves 100× memory and storage
- Produces portable 161MB adapter file

**Does NOT determine:**
- What the model learns
- Training objectives
- Loss functions

---

### The Three Losses (The OBJECTIVES)

**Role:** WHAT we want the model to learn

1. **Distillation Loss (KL divergence):**
   - Learn teacher's probability distributions
   - Captures "dark knowledge"
   - Soft targets with temperature

2. **Ground Truth Loss (Cross-entropy):**
   - Learn the correct answer
   - Hard targets from training data
   - Ensures factual correctness

3. **Total Loss:**
   - Balanced combination (50/50 in your project)
   - Guides gradient updates
   - Tells LoRA adapters which direction to change

---

### How They Work Together

```
┌──────────────────────────────────────────────────────────┐
│                   Training Loop                          │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Forward Pass: Student (Base + LoRA) predicts            │
│  Forward Pass: Teacher (frozen) predicts                 │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Compute Losses:                                         │
│    • Distillation Loss = KL(teacher || student)          │
│    • Ground Truth Loss = CE(student, target)             │
│    • Total Loss = 0.5 × each                             │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Backpropagation: Compute gradients                      │
│    • Gradients for LoRA adapters (A, B matrices)         │
│    • No gradients for base model (frozen)                │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Update: Only LoRA parameters change                     │
│    • A_new = A_old - learning_rate × gradient_A          │
│    • B_new = B_old - learning_rate × gradient_B          │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
                  Repeat for next batch
```

**Key insight:**
- **Losses** tell us the model is wrong and by how much
- **Gradients** tell us which direction to move weights
- **LoRA** tells us which weights are allowed to move
- **Optimizer** decides how far to move them

---

## Practical Implications

### Why This Design is Brilliant

1. **Memory efficient:** Only update 1% of parameters (LoRA)
2. **Knowledge transfer:** Learn from teacher's soft targets (Distillation loss)
3. **Stay grounded:** Maintain correctness (Ground truth loss)
4. **Portable:** 161MB adapter vs 16GB full model
5. **Fast training:** Fewer parameters = faster backprop

### Trade-offs You Can Adjust

**LoRA rank (r):**
- `r=8`: Smaller adapters (80MB), less expressive, faster training
- `r=16`: Medium (161MB), good balance ← **your project**
- `r=32`: Larger (322MB), more expressive, slower training

**Alpha (loss weighting):**
- `alpha=0.7`: Trust teacher more (70% distillation, 30% ground truth)
- `alpha=0.5`: Balanced ← **your project**
- `alpha=0.3`: Trust data more (30% distillation, 70% ground truth)

**Temperature:**
- `T=1.0`: Standard softmax ← **your project**
- `T=2.0`: Softer distributions (more information from teacher)
- `T=4.0`: Very soft (useful for difficult tasks)

---

**Last Updated:** November 4, 2024
