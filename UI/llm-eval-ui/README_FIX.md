# HumanEval "Some problems are not attempted" - SOLUTION

## Problem
The error `AssertionError: Some problems are not attempted` occurred when testing individual HumanEval problems because the `evaluate_functional_correctness()` function requires **ALL 164 problems** to be in the JSONL file.

## Root Causes

### 1. Missing Problems Requirement
The HumanEval evaluation framework checks:
```python
assert len(completion_id) == len(problems), "Some problems are not attempted."
```

This means you must provide a completion for every problem in the dataset, even if you're only testing one.

### 2. Missing Indentation
Model completions often lack proper indentation. HumanEval expects:
- **Prompt**: Contains `from typing...`, `def function(...):`, and docstring
- **Completion**: Contains ONLY the function body with **4-space indentation**

When combined: `prompt + completion = valid Python`

Without indentation:
```python
def has_close_elements(...):
    """docstring"""
for idx in numbers:  # ❌ Not indented! Invalid Python!
```

With indentation:
```python
def has_close_elements(...):
    """docstring"""
    for idx in numbers:  # ✅ Properly indented!
```

## Solution

### File Structure
```
llm-eval-ui/
├── app.py              # Main Streamlit app
├── eval_helper.py      # NEW: Handles HumanEval quirks
├── config.py           # Model configurations
└── requirements.txt    # Dependencies
```

### Key Changes

#### 1. New File: `eval_helper.py`
Location: `/mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui/eval_helper.py`

This module handles the "all 164 problems" requirement:

```python
def test_single_problem(problem_id: str, completions: list) -> dict:
    """
    Test solutions for a single HumanEval problem.

    Workaround: Adds placeholder empty completions for all other problems
    to satisfy the framework's requirement.
    """
    # For each completion, create a full 164-problem JSONL file:
    # - Target problem: actual completion
    # - Other 163 problems: empty placeholder (will fail, but we ignore)

    # Test each completion separately
    # Extract only the result for the target problem
    # Return aggregated results
```

#### 2. Updated: `app.py`

**Import the helper:**
```python
from eval_helper import test_single_problem
```

**Modified `test_solutions()` function:**
- Extracts and cleans all completions (steps 1-5)
- **Automatically adds 4-space indentation if missing** (step 5)
- Calls `test_single_problem()` which handles the 164-problem requirement
- Returns results with pass@k metrics

**Indentation fix (app.py lines 501-517):**
```python
# Step 5: CRITICAL - Ensure completion has proper indentation
if completion:
    current_lines = completion.split('\n')
    first_code_line = next((line for line in current_lines if line.strip()), None)
    if first_code_line and not first_code_line.startswith((' ', '\t')):
        # No indentation! Add 4 spaces to every non-empty line
        indented_lines = []
        for line in current_lines:
            if line.strip():
                indented_lines.append('    ' + line)
            else:
                indented_lines.append(line)
        completion = '\n'.join(indented_lines)
```

### How It Works

1. **User selects problem** (e.g., HumanEval/0)
2. **Model generates N solutions** (e.g., 10 solutions)
3. **Extract & clean completions:**
   - Remove docstring regeneration
   - Stop at markdown fences or examples
   - **Add 4-space indentation if missing**
4. **Test each completion separately:**
   - Create JSONL with all 164 problems
   - Target problem: actual completion
   - Other 163: empty placeholder
5. **Run HumanEval evaluation**
6. **Extract result for target problem only**
7. **Calculate pass@k metrics**

### Benefits

✅ Tests individual problems without needing all 164
✅ Automatically fixes indentation issues
✅ Provides detailed debugging info (char count, line count, indentation status)
✅ Works with existing Streamlit UI
✅ Logs to MLflow correctly

### Testing

Run the test scripts to verify:

```bash
cd /mnt/bst/achoi13A100/ptuemler/LLM-Compression-Trade-off-Evaluator/UI/llm-eval-ui

# Test indentation fix
python test_extraction.py

# Test HumanEval format
python test_humaneval_format.py

# Test eval_helper with canonical solution
python test_eval_helper.py
```

All tests should pass! ✅

### Usage in Streamlit App

1. Start the app: `./run.sh`
2. Select a model (Student, Distilled, or Teacher)
3. Choose a HumanEval problem
4. Generate solutions
5. View results with pass@k metrics
6. Check debugging info if needed

The app will now:
- Automatically fix indentation
- Successfully evaluate solutions
- Show detailed metrics in MLflow
- Display complexity scores (1-10) and routing recommendations

## Summary

The "Some problems are not attempted" error had TWO causes:
1. **Framework requirement**: Must provide all 164 problems → Fixed with `eval_helper.py`
2. **Missing indentation**: Completions need 4 spaces → Fixed in `test_solutions()` step 5

Both issues are now resolved! 🎉
