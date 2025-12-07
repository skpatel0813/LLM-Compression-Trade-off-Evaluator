LLM Evaluation UI - Complete Summary
Project Overview
Built an interactive Streamlit + MLflow web application for testing LLM models on the HumanEval benchmark, specifically for evaluating knowledge distillation from a 70B teacher model to an 8B student model.
Location
/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui/
Project Structure
llm-eval-ui/
├── app.py                      # Main Streamlit application (~820 lines)
├── eval_helper.py              # HumanEval evaluation wrapper (handles 164-problem requirement)
├── config.py                   # Model paths and MLflow configuration
├── requirements.txt            # Python dependencies
├── run.sh                      # Launch script for Streamlit app
├── run_mlflow.sh              # Launch script for MLflow UI
├── README_FIX.md              # Detailed documentation of bug fixes
├── test_extraction.py          # Test script for code extraction logic
├── test_humaneval_format.py   # Test script for HumanEval format validation
└── test_eval_helper.py        # Test script for eval_helper module
Source Code Files
1. app.py - Main Application
Purpose: Interactive Streamlit dashboard for LLM evaluation Key Functions:
load_model(model_name) (lines 43-80): Loads Student/Distilled/Teacher models with caching
generate_solution() (lines 83-130): Generates code completions with configurable temperature/max_tokens
extract_code_from_completion() (lines 133-386): Extracts Python functions from model outputs (handles multiple edge cases)
estimate_query_complexity() (lines 388-428): Token-based complexity scoring (1-10 scale) and routing recommendations
calculate_pass_at_k() (lines 430-447): Pass@k metric calculation
test_solutions() (lines 450-572):
Extracts completions from model outputs
Removes docstring regeneration
Automatically adds 4-space indentation if missing
Calls eval_helper.test_single_problem()
Returns pass@k metrics
UI Features:
Model selection (Student 8B, Distilled 8B+LoRA, Teacher 70B)
HumanEval problem browser (164 problems)
Adjustable generation parameters (temperature: 0.0-1.0, max tokens: 128-1024)
Number of solutions (1-20 with pass@k calculation)
Real-time generation progress
Detailed debugging output (character count, line count, indentation status)
MLflow experiment logging
Critical Fix (lines 501-517):
# Step 5: Ensure proper indentation (HumanEval requirement)
if completion:
    current_lines = completion.split('\n')
    first_code_line = next((line for line in current_lines if line.strip()), None)
    if first_code_line and not first_code_line.startswith((' ', '\t')):
        # No indentation! Add 4 spaces to every non-empty line
        for line in current_lines:
            if line.strip():
                indented_lines.append('    ' + line)
2. eval_helper.py - Evaluation Wrapper
Purpose: Handles HumanEval's requirement that ALL 164 problems must be in the JSONL file Key Functions:
test_single_problem(problem_id, completions) (lines 13-64):
Tests multiple completions for a single HumanEval problem
Aggregates results from individual completion tests
Returns: {success, num_correct, num_total, detailed_results}
_test_single_completion(problem_id, completion, all_problems) (lines 67-137):
Creates JSONL with all 164 problems
Target problem gets actual completion
Other 163 problems get empty placeholders
Runs evaluate_functional_correctness()
Extracts only the result for target problem
Returns: {passed, result}
Why This Was Needed:
# HumanEval framework checks:
assert len(completion_id) == len(problems), "Some problems are not attempted."
Without this helper, testing a single problem would fail with the error above.
3. config.py - Configuration
Purpose: Centralized model paths and settings Contents:
MODEL_CONFIGS = {
    "Student (8B)": {
        "path": "/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/models/student",
        "type": "base"
    },
    "Distilled (8B + LoRA)": {
        "path": "/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/models/student",
        "adapter_path": "/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/models/distilled_adapter",
        "type": "lora"
    },
    "Teacher (70B)": {
        "path": "/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/models/teacher",
        "type": "base"
    }
}

DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.2,
    "max_new_tokens": 256,
    "do_sample": True,
    "top_p": 0.95
}

MLFLOW_TRACKING_URI = "./mlruns"
RESULTS_DIR = "./results"
4. requirements.txt - Dependencies
Purpose: Python package dependencies Contents:
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
accelerate>=0.24.0
human-eval>=1.0.0
mlflow>=2.8.0
plotly>=5.17.0
pandas>=2.0.0
tiktoken>=0.5.0
numpy>=1.24.0
datasets>=2.14.0
Installation Location: llm311 conda environment
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/

How Requirements Were Installed:
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/pip install -r requirements.txt
5. run.sh - Streamlit Launch Script
#!/bin/bash
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address localhost
6. run_mlflow.sh - MLflow UI Launch Script
#!/bin/bash
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui
/mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/bin/mlflow ui \
    --backend-store-uri ./mlruns \
    --port 5000
Key Issues Resolved
Issue 1: "Some problems are not attempted"
Root Cause: HumanEval's evaluate_functional_correctness() requires ALL 164 problems in JSONL Solution: Created eval_helper.py that:
Creates 164-problem JSONL for each test
Fills untested problems with empty placeholders
Extracts only target problem's result
Evidence:
# Before fix:
AssertionError: Some problems are not attempted.

# After fix:
✅ SUCCESS: Canonical solution passed!
Issue 2: Missing Indentation
Root Cause: Models generate code without leading spaces, but HumanEval expects prompt + completion = valid Python Expected Format:
# Prompt (from HumanEval):
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """docstring"""

# Completion (must be indented):
    for idx, elem in enumerate(numbers):
        # ...
    return False
Solution: Automatic indentation detection and correction (app.py lines 501-517)
Issue 3: MLflow Metric Names
Root Cause: MLflow rejects @ symbol in metric names Solution: Replace @ with _at_ when logging
# pass@1 → pass_at_1
# pass@5 → pass_at_5
# pass@10 → pass_at_10
Source Code from Parent Project
The UI integrates with models from the main project:
Model Locations
/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/
├── models/
│   ├── student/              # 8B base model
│   ├── distilled_adapter/    # 161MB LoRA adapters
│   └── teacher/              # 70B teacher model
├── src/
│   ├── routing_system.py     # Complexity-based routing logic (source of 1-10 scoring)
│   └── eval_humaneval_*.py   # Original evaluation scripts
└── UI/
    └── llm-eval-ui/          # Our new UI app
Routing Logic Reference
From routing_system.py (complexity scoring basis):
# Complexity 1-6 (≤150 tokens): Simple → Use Distilled (8B)
# Complexity 7-10 (>150 tokens): Complex → Use Teacher (70B)
Environment Setup
Conda Environment
Name: llm311
Python: 3.11
Location: /mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/
Key Packages Installed
Streamlit: Web UI framework
Transformers: Hugging Face model loading
PEFT: LoRA adapter loading
human-eval: HumanEval benchmark framework
MLflow: Experiment tracking
tiktoken: Token counting for complexity estimation
PyTorch: Deep learning framework
Usage
1. Start Streamlit App
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui
./run.sh
Access at: http://localhost:8501
2. Start MLflow UI (optional)
./run_mlflow.sh
Access at: http://localhost:5000
3. Workflow
Select model (Student/Distilled/Teacher)
Choose HumanEval problem (1-164)
Set generation parameters
Generate solutions
View pass@k metrics and complexity scores
Check MLflow for logged experiments
Testing Scripts
test_extraction.py
Tests code extraction and indentation logic
python test_extraction.py
# Output: ✅ SUCCESS: Completion starts with 4 spaces
test_humaneval_format.py
Validates HumanEval format (prompt + completion)
python test_humaneval_format.py
# Output: ✅ Valid Python syntax!
test_eval_helper.py
Tests eval_helper with canonical solutions
python test_eval_helper.py
# Output: ✅ SUCCESS! Canonical solution passed!
Documentation
README_FIX.md
Comprehensive documentation covering:
Problem analysis
Root causes (all 3 issues)
Solutions with code snippets
File structure
Testing procedures
Usage instructions
Summary
What We Built: Interactive Streamlit UI for testing LLM models on HumanEval with MLflow experiment tracking Key Features:
✅ Test 3 models (Student 8B, Distilled 8B+LoRA, Teacher 70B)
✅ Generate multiple solutions per problem
✅ Automatic code extraction and indentation fixing
✅ Pass@k metrics calculation (pass@1, pass@5, pass@10)
✅ Complexity scoring (1-10) and routing recommendations
✅ MLflow experiment logging
✅ Detailed debugging output
Issues Resolved:
HumanEval 164-problem requirement → eval_helper.py
Missing indentation → Automatic fix in app.py
MLflow metric names → @ → _at_ conversion
Environment: llm311 conda environment at /mnt/bst/achoi13A100/ptuemler/miniconda3/envs/llm311/ All tests passing ✅