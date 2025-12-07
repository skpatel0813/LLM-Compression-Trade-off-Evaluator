"""
Configuration file for LLM Eval UI
Points to resources in the parent training project
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # LLM-Compression-Trade-off-Evaluator/
UI_ROOT = Path(__file__).parent.parent              # UI/

# Model paths (cached in system)
TEACHER_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
STUDENT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# LoRA adapters path
LORA_ADAPTERS_PATH = str(PROJECT_ROOT / "outputs" / "llama31_8b_kd_lora" / "lora")

# MLflow tracking
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"

# Results directory (for loading past evaluation results)
RESULTS_DIR = str(PROJECT_ROOT / "results")

# Model configurations
MODEL_CONFIGS = {
    "Student (8B Base)": {
        "model_name": STUDENT_MODEL,
        "load_in_8bit": False,
        "lora_path": None,
        "description": "Base Llama-3.1-8B model without any fine-tuning"
    },
    "Distilled (8B + LoRA)": {
        "model_name": STUDENT_MODEL,
        "load_in_8bit": False,
        "lora_path": LORA_ADAPTERS_PATH,
        "description": "8B model with trained LoRA adapters from knowledge distillation"
    },
    "Teacher (70B - 8bit)": {
        "model_name": TEACHER_MODEL,
        "load_in_8bit": True,
        "lora_path": None,
        "description": "Llama-3.1-70B model quantized to 8-bit"
    }
}

# Generation parameters
DEFAULT_GENERATION_PARAMS = {
    "max_new_tokens": 512,
    "temperature": 0.01,  # Nearly greedy for code generation
    "do_sample": True,
    "top_p": 0.95,
    "num_return_sequences": 1
}
