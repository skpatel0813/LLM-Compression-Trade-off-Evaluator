"""
LLM Evaluation UI - Streamlit App
Interactive interface for testing and comparing LLM models on HumanEval
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
import mlflow
import json
from pathlib import Path
import time
import tempfile
import numpy as np
import tiktoken

# Import config
from config import (
    MODEL_CONFIGS,
    DEFAULT_GENERATION_PARAMS,
    MLFLOW_TRACKING_URI,
    RESULTS_DIR
)

# Import evaluation helper
from eval_helper import test_single_problem

# Page config
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or get experiment
try:
    experiment = mlflow.get_experiment_by_name("November experiment")
    if experiment is None:
        experiment_id = mlflow.create_experiment("November experiment")
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment("November experiment")
except Exception as e:
    st.sidebar.warning(f"MLflow setup warning: {e}")

# Title
st.title("🤖 LLM Compression Evaluation Dashboard")
st.markdown("Interactive testing of Student, Teacher, and Distilled models on HumanEval benchmark")

# Sidebar - Model Selection
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_CONFIGS.keys()),
    help="Choose which model to use for code generation"
)

# Show model info
model_info = MODEL_CONFIGS[model_choice]
st.sidebar.info(f"**Description:** {model_info['description']}")

# Generation parameters
st.sidebar.subheader("Generation Parameters")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=DEFAULT_GENERATION_PARAMS["temperature"],
    step=0.01,
    help="Lower = more deterministic, Higher = more creative"
)

max_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=128,
    max_value=1024,
    value=DEFAULT_GENERATION_PARAMS["max_new_tokens"],
    step=128,
    help="Maximum length of generated code"
)

num_samples = st.sidebar.number_input(
    "Number of Samples",
    min_value=1,
    max_value=10,
    value=1,
    help="Generate multiple solutions (for pass@k metrics)"
)


# Cache model loading
@st.cache_resource
def load_model(model_name, load_in_8bit, lora_path):
    """Load model with optional LoRA adapters"""

    with st.spinner(f"Loading {model_name}..."):
        # Load base model
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

        # Load LoRA adapters if specified
        if lora_path:
            st.info(f"Loading LoRA adapters from: {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


# Cache HumanEval problems
@st.cache_data
def load_humaneval():
    """Load HumanEval benchmark problems"""
    return read_problems()


def extract_code_from_completion(completion: str, prompt: str = "") -> str:
    """
    Extract clean Python code from model completion.
    Handles various output formats from LLMs.
    """
    if not completion:
        return ""

    original_completion = completion
    completion = completion.strip()

    # Remove markdown code fences
    # Case 1: Starts with ```python or ```
    if completion.startswith("```python"):
        start = len("```python")
        end = completion.find("```", start)
        if end != -1:
            completion = completion[start:end].strip()
    elif completion.startswith("```"):
        start = 3
        end = completion.find("```", start)
        if end != -1:
            completion = completion[start:end].strip()
    # Case 2: ```python appears later in the completion
    elif "```python" in completion:
        start = completion.find("```python") + len("```python")
        end = completion.find("```", start)
        if end != -1:
            completion = completion[start:end].strip()
        else:
            # No closing fence, take everything after ```python
            completion = completion[start:].strip()
    # Case 3: ``` appears as a terminator (stop there)
    elif "```" in completion:
        completion = completion.split("```")[0].strip()

    # Remove common prefixes
    for prefix in ["assistant\n", "Assistant\n", "ASSISTANT\n", "assistant:", "Assistant:"]:
        if completion.startswith(prefix):
            completion = completion[len(prefix):].strip()

    # Find ALL function definitions
    lines = completion.split('\n')
    def_positions = []

    # Extract target function name from prompt if available
    target_func_name = None
    if prompt:
        for line in prompt.split('\n'):
            if line.strip().startswith('def '):
                # Extract function name: "def func_name(" -> "func_name"
                func_def = line.strip()
                if '(' in func_def:
                    target_func_name = func_def.split('def ')[1].split('(')[0].strip()
                    break

    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            def_positions.append(i)

    if not def_positions:
        # No 'def' found in completion - check if prompt has it
        # Sometimes model only generates the body
        if prompt:
            prompt_lines = prompt.split('\n')
            for i, line in enumerate(prompt_lines):
                if line.strip().startswith('def '):
                    # Found function signature in prompt!
                    # Extract it and combine with completion
                    func_signature = line
                    # Find the proper indentation level from prompt
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * (indent + 4)  # Body should be indented 4 more

                    # Re-indent completion to match expected body indentation
                    # BUT stop at markdown code fences or example sections
                    completion_lines = [func_signature]
                    found_code = False  # Track if we've found actual code (not docstring)

                    for comp_line in lines:
                        stripped = comp_line.strip()

                        # Stop at markdown fence or "Example" sections
                        if stripped.startswith('```') or \
                           stripped.lower().startswith('example'):
                            break

                        # Check if this is actual Python code (not docstring continuation)
                        is_code = (stripped.startswith('#') or
                                  stripped.startswith('"""') or
                                  stripped.startswith("'''") or
                                  any(stripped.startswith(kw) for kw in
                                      ['return ', 'if ', 'for ', 'while ', 'def ', 'class ',
                                       'import ', 'from ', 'with ', 'try:', 'except', 'raise ']) or
                                  (found_code and stripped) or  # After we find code, include everything
                                  '=' in stripped or  # Variable assignments
                                  stripped.endswith(':') or  # Control flow
                                  (comp_line.startswith('    ') and stripped))  # Indented content

                        if is_code:
                            found_code = True

                        if found_code or not stripped:  # Include code lines and empty lines
                            if stripped:  # Non-empty line
                                # Add proper indentation if not already there
                                if not comp_line.startswith(' '):
                                    completion_lines.append(indent_str + comp_line)
                                else:
                                    completion_lines.append(comp_line)
                            elif completion_lines:  # Empty lines after we've started
                                completion_lines.append('')

                    return '\n'.join(completion_lines).strip()

        # Still no function found - return as-is if it has code-like content
        if completion and any(line.strip() for line in lines):
            return completion
        return ""

    # Find the correct function to extract
    # If we have a target function name from the prompt, look for it
    start_idx = def_positions[-1]  # Default to last function

    if target_func_name:
        # Try to find the function that matches the target name
        matching_positions = []
        for pos in def_positions:
            line = lines[pos].strip()
            # Check if this line defines the target function
            if line.startswith(f'def {target_func_name}('):
                matching_positions.append(pos)

        # Use the LAST occurrence of the target function (in case it's regenerated)
        if matching_positions:
            start_idx = matching_positions[-1]

    # Extract from the selected function onwards
    code_lines = []
    in_function = False
    indent_level = None
    in_docstring = False
    docstring_char = None

    for i in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Start of function
        if i == start_idx and stripped.startswith('def '):
            in_function = True
            code_lines.append(line)
            # Determine base indentation
            indent_level = len(line) - len(line.lstrip())
            continue

        if not in_function:
            continue

        # Handle docstrings (they can span multiple lines)
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = '"""' if stripped.startswith('"""') else "'''"
                code_lines.append(line)
                # Check if docstring ends on same line
                if stripped.count(docstring_char) >= 2:
                    in_docstring = False
                else:
                    in_docstring = True
                continue
        else:
            # Inside multi-line docstring
            code_lines.append(line)
            if docstring_char in stripped:
                in_docstring = False
            continue

        # Inside function
        if not stripped:
            # Empty line - include it
            code_lines.append(line)
            continue

        # Calculate current line indentation
        current_indent = len(line) - len(line.lstrip())

        # Check if line is part of function body (including nested functions)
        # It should be indented more than the def line
        if current_indent > indent_level:
            code_lines.append(line)
        elif stripped.startswith('#'):
            # Comments at any level
            code_lines.append(line)
        else:
            # Reached end of function (same or less indentation, non-empty line that's not a comment)
            break

    result = '\n'.join(code_lines).strip()

    # Validate we have a proper function
    if not result:
        # Fallback: return the whole completion if it has code
        return original_completion if 'def ' in original_completion else ""

    # Check we have actual implementation (not just def line)
    result_lines = [l for l in result.split('\n') if l.strip() and not l.strip().startswith('#')]
    if len(result_lines) < 2:
        # Only has def line, no body - try harder fallback
        # Maybe the indentation detection failed, try returning just code-like content
        if original_completion.strip():
            # Return all lines that look like code
            all_lines = original_completion.split('\n')

            # If we have a target function name, try to find it
            if target_func_name:
                for i in range(len(all_lines)-1, -1, -1):
                    line = all_lines[i].strip()
                    if line.startswith(f'def {target_func_name}('):
                        # Found the target function, extract everything from here
                        # until we find a line with equal or less indentation
                        base_indent = len(all_lines[i]) - len(all_lines[i].lstrip())
                        func_lines = [all_lines[i]]
                        for j in range(i + 1, len(all_lines)):
                            curr_line = all_lines[j]
                            curr_stripped = curr_line.strip()
                            if not curr_stripped:  # Empty line
                                func_lines.append(curr_line)
                                continue
                            curr_indent = len(curr_line) - len(curr_line.lstrip())
                            if curr_indent > base_indent or curr_stripped.startswith('#'):
                                func_lines.append(curr_line)
                            elif curr_stripped.startswith('```') or curr_stripped.lower().startswith('example'):
                                break  # Hit markdown or examples
                            else:
                                break  # Hit dedented line
                        result_fallback = '\n'.join(func_lines).strip()
                        if result_fallback and len([l for l in result_fallback.split('\n') if l.strip()]) > 1:
                            return result_fallback

            # Generic fallback: find the last def and take everything
            for i in range(len(all_lines)-1, -1, -1):
                if 'def ' in all_lines[i] and '(' in all_lines[i]:
                    return '\n'.join(all_lines[i:]).strip()
            return original_completion.strip()
        return ""

    return result


def estimate_query_complexity(prompt: str) -> dict:
    """
    Estimate query complexity based on token count.
    Returns complexity score (1-10) and routing decision.

    Routing strategy (from routing_system.py):
    - Complexity 1-6 (≤150 tokens): Simple → Use Distilled (8B)
    - Complexity 7-10 (>150 tokens): Complex → Use Teacher (70B)
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(prompt)
        token_count = len(tokens)

        # Map token count to complexity (1-10)
        if token_count <= 150:
            # Simple/Medium: map to 1-6
            complexity = max(1, min(6, token_count // 25))
            routing = "Distilled (8B)"
            category = "Simple/Medium"
        else:
            # Complex: map to 7-10
            complexity = 7 + min(3, (token_count - 151) // 50)
            routing = "Teacher (70B)"
            category = "Complex"

        return {
            'token_count': token_count,
            'complexity': complexity,
            'routing': routing,
            'category': category
        }
    except Exception as e:
        return {
            'token_count': 0,
            'complexity': 5,
            'routing': 'Unknown',
            'category': 'Unknown',
            'error': str(e)
        }


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        Probability that at least one of k samples is correct
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def test_solutions(problem_id: str, problem: dict, solutions: list) -> dict:
    """
    Test generated solutions against HumanEval test cases.

    Returns dict with test results and pass@k metrics.

    IMPORTANT: HumanEval expects completion to be ONLY the function body,
    not the full function. The framework does: prompt + completion = full code.
    """
    # Extract and clean all completions
    extracted_codes = []
    completions = []

    for i, sol in enumerate(solutions):
        # Raw completion from model
        completion = sol["solution"].strip()

        # Step 1: Remove everything before closing """ if present
        # (Model often regenerates docstring)
        if '"""' in completion:
            parts = completion.split('"""')
            # Take everything AFTER the first """ (which closes the docstring)
            if len(parts) > 1:
                completion = '"""'.join(parts[1:])

        # Step 2: Stop at ``` (end of code block)
        if "```" in completion:
            completion = completion.split("```")[0]

        # Step 3: Stop at "Example" sections
        if "example" in completion.lower():
            lines = completion.split('\n')
            code_lines = []
            for line in lines:
                if line.strip().lower().startswith('example'):
                    break
                code_lines.append(line)
            completion = '\n'.join(code_lines)

        # Step 4: Clean up but preserve leading whitespace (indentation is critical!)
        # Remove leading/trailing blank lines, but keep indentation
        lines = completion.split('\n')
        # Remove leading empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        completion = '\n'.join(lines)

        # Step 5: CRITICAL - Ensure completion has proper indentation
        # HumanEval expects function body to be indented (4 spaces typically)
        if completion:
            # Re-split to check current state
            current_lines = completion.split('\n')
            # Check if first non-empty line has indentation
            first_code_line = next((line for line in current_lines if line.strip()), None)
            if first_code_line and not first_code_line.startswith((' ', '\t')):
                # No indentation! Need to add it
                # Add 4 spaces to every non-empty line
                indented_lines = []
                for line in current_lines:
                    if line.strip():  # Non-empty line
                        indented_lines.append('    ' + line)
                    else:  # Empty line
                        indented_lines.append(line)
                completion = '\n'.join(indented_lines)

        # Store for debugging with detailed info
        final_lines = completion.split('\n') if completion else []
        extracted_codes.append({
            'completion': completion,
            'length': len(completion),
            'first_line': repr(final_lines[0]) if final_lines else '',
            'has_indentation': any(line.startswith(' ') or line.startswith('\t') for line in final_lines if line.strip()),
            'line_count': len([l for l in final_lines if l.strip()])
        })

        completions.append(completion)

    # Use the helper function to test all completions
    # This handles the "all 164 problems" requirement internally
    try:
        result = test_single_problem(problem_id, completions)

        if result['success']:
            # Calculate pass@k metrics
            num_correct = result['num_correct']
            num_total = result['num_total']

            pass_at_k = {}
            for k in [1, 5, 10]:
                if k <= num_total:
                    pass_at_k[f'pass@{k}'] = calculate_pass_at_k(num_total, num_correct, k)

            return {
                'success': True,
                'num_correct': num_correct,
                'num_total': num_total,
                'pass_at_k': pass_at_k,
                'detailed_results': result['detailed_results'],
                'extracted_codes': extracted_codes
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'num_correct': 0,
                'num_total': len(solutions),
                'pass_at_k': {},
                'extracted_codes': extracted_codes
            }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'num_correct': 0,
            'num_total': len(solutions),
            'pass_at_k': {},
            'extracted_codes': extracted_codes
        }


# Load HumanEval
try:
    problems = load_humaneval()
    st.sidebar.success(f"✅ Loaded {len(problems)} HumanEval problems")
except Exception as e:
    st.sidebar.error(f"❌ Error loading HumanEval: {e}")
    st.stop()


# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧪 Interactive Testing",
    "📊 Model Comparison",
    "📈 Past Results",
    "ℹ️ About"
])


# Tab 1: Interactive Testing
with tab1:
    st.header("Interactive Code Generation")

    # Problem selection
    col1, col2 = st.columns([1, 3])

    with col1:
        problem_id = st.selectbox(
            "Select Problem",
            sorted(list(problems.keys())),
            help="Choose a HumanEval problem to test"
        )

    with col2:
        st.metric("Problem ID", problem_id)

    # Display problem
    st.subheader("Problem Statement")
    problem = problems[problem_id]

    # Calculate complexity
    complexity_info = estimate_query_complexity(problem["prompt"])

    # Show complexity metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tokens", complexity_info['token_count'])
    with col2:
        st.metric("Complexity", f"{complexity_info['complexity']}/10")
    with col3:
        st.metric("Category", complexity_info['category'])
    with col4:
        recommended = "✅" if complexity_info['routing'] in model_choice else "⚠️"
        st.metric("Recommended", f"{recommended} {complexity_info['routing']}")

    st.code(problem["prompt"], language="python")

    # Generate button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        generate_btn = st.button("🚀 Generate Solution", type="primary", use_container_width=True)
    with col2:
        log_to_mlflow = st.checkbox("Log to MLflow", value=True)

    if generate_btn:
        try:
            # Load model
            model, tokenizer = load_model(
                model_info["model_name"],
                model_info["load_in_8bit"],
                model_info["lora_path"]
            )

            # Prepare input
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. Write clean, correct code."},
                {"role": "user", "content": problem["prompt"]}
            ]

            input_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            # Generate
            st.subheader("Generated Solutions")
            solutions = []

            progress_bar = st.progress(0)

            for i in range(num_samples):
                with st.spinner(f"Generating solution {i+1}/{num_samples}..."):
                    start_time = time.time()

                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    generation_time = time.time() - start_time

                    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Extract just the generated part (remove prompt)
                    generated_part = completion[len(input_text):]

                    solutions.append({
                        "solution": generated_part,
                        "time": generation_time
                    })

                    # Display solution
                    with st.expander(f"Solution {i+1} (Generated in {generation_time:.2f}s)", expanded=(i==0)):
                        st.code(generated_part, language="python")

                    progress_bar.progress((i + 1) / num_samples)

            st.success(f"✅ Generated {num_samples} solution(s)")

            # Test solutions against test cases
            st.subheader("🧪 Testing Solutions")
            with st.spinner("Running test cases..."):
                test_results = test_solutions(problem_id, problem, solutions)

            if test_results['success']:
                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Passed",
                        f"{test_results['num_correct']}/{test_results['num_total']}",
                        delta=f"{test_results['num_correct']/test_results['num_total']*100:.1f}%"
                    )

                # Display pass@k metrics
                if test_results['pass_at_k']:
                    with col2:
                        if 'pass@1' in test_results['pass_at_k']:
                            st.metric("Pass@1", f"{test_results['pass_at_k']['pass@1']*100:.1f}%")
                    with col3:
                        if 'pass@5' in test_results['pass_at_k']:
                            st.metric("Pass@5", f"{test_results['pass_at_k']['pass@5']*100:.1f}%")

                # Show detailed results
                with st.expander("Detailed Test Results"):
                    for i, result in enumerate(test_results['detailed_results']):
                        status = "✅ PASSED" if result.get('passed') else "❌ FAILED"
                        st.write(f"**Solution {i+1}:** {status}")

                        # Show extracted code
                        if 'completion' in result:
                            with st.expander(f"Extracted Code (Solution {i+1})", expanded=False):
                                st.code(result['completion'], language='python')

                        # Show error if failed
                        if not result.get('passed') and 'result' in result:
                            st.error(f"Error: {result['result']}")

                st.success("✅ Testing complete")
            else:
                st.error(f"❌ Testing failed: {test_results.get('error', 'Unknown error')}")

                # Show what was extracted for debugging
                with st.expander("Debug: Extracted Code", expanded=True):
                    extracted_codes = test_results.get('extracted_codes', [])

                    for i, sol in enumerate(solutions):
                        st.write(f"**Solution {i+1}:**")

                        # Show what was extracted
                        if i < len(extracted_codes):
                            extracted_info = extracted_codes[i]
                            # Check if it's the new dict format or old string format
                            if isinstance(extracted_info, dict):
                                extracted = extracted_info['completion']
                                st.success(f"✅ Extracted {extracted_info['length']} characters, {extracted_info.get('line_count', 0)} lines")
                                st.info(f"First line (with quotes to show spaces): {extracted_info['first_line']}")
                                st.info(f"Has indentation: {extracted_info['has_indentation']}")
                            else:
                                extracted = extracted_info
                                st.success(f"✅ Extracted {len(extracted)} characters")
                        else:
                            extracted = extract_code_from_completion(sol["solution"], problem["prompt"])

                        if extracted:
                            st.code(extracted, language='python')

                            # Show more debugging info
                            with st.expander("Full completion (for debugging)", expanded=False):
                                st.code(sol["solution"], language='text')
                        else:
                            st.error("❌ No code extracted!")
                            st.write("**Original completion:**")
                            st.code(sol["solution"], language='text')

                            st.write("**Problem prompt (first 500 chars):**")
                            st.code(problem["prompt"][:500], language='python')

                test_results['pass_at_k'] = {}

            # Log to MLflow
            if log_to_mlflow:
                with mlflow.start_run(run_name=f"{model_choice}_{problem_id}"):
                    mlflow.log_param("model", model_choice)
                    mlflow.log_param("problem_id", problem_id)
                    mlflow.log_param("temperature", temperature)
                    mlflow.log_param("max_tokens", max_tokens)
                    mlflow.log_param("num_samples", num_samples)

                    # Log complexity metrics
                    mlflow.log_param("token_count", complexity_info['token_count'])
                    mlflow.log_param("complexity", complexity_info['complexity'])
                    mlflow.log_param("category", complexity_info['category'])
                    mlflow.log_param("recommended_model", complexity_info['routing'])

                    for i, sol in enumerate(solutions):
                        mlflow.log_text(sol["solution"], f"solution_{i+1}.py")
                        mlflow.log_metric(f"generation_time_{i+1}", sol["time"])

                    mlflow.log_metric("avg_generation_time", sum(s["time"] for s in solutions) / len(solutions))

                    # Log test results
                    mlflow.log_metric("num_correct", test_results['num_correct'])
                    mlflow.log_metric("num_total", test_results['num_total'])
                    mlflow.log_metric("pass_rate", test_results['num_correct'] / test_results['num_total'])

                    # Log pass@k metrics (replace @ with _at_ for MLflow compatibility)
                    for k_name, k_value in test_results.get('pass_at_k', {}).items():
                        # MLflow doesn't allow @ in metric names
                        mlflow_metric_name = k_name.replace('@', '_at_')
                        mlflow.log_metric(mlflow_metric_name, k_value)

                st.success("✅ Logged to MLflow")

        except Exception as e:
            st.error(f"❌ Error during generation: {e}")
            import traceback
            st.code(traceback.format_exc())


# Tab 2: Model Comparison
with tab2:
    st.header("Side-by-Side Model Comparison")
    st.info("🚧 Coming soon: Compare outputs from all 3 models on the same problem")

    # Placeholder for comparison feature
    st.markdown("""
    **Planned features:**
    - Generate from all 3 models simultaneously
    - Display outputs side-by-side
    - Run against test cases
    - Show pass/fail for each model
    - Compare generation times and GPU usage
    """)


# Tab 3: Past Results
with tab3:
    st.header("Past Evaluation Results")

    try:
        results_path = Path(RESULTS_DIR)
        metric_files = list(results_path.glob("*.metrics.json"))

        if metric_files:
            st.success(f"Found {len(metric_files)} evaluation results")

            # Load and display results
            results_data = []
            for metric_file in metric_files:
                with open(metric_file) as f:
                    data = json.load(f)
                    data["filename"] = metric_file.name
                    results_data.append(data)

            # Select result to view
            result_choice = st.selectbox(
                "Select Evaluation Run",
                [r["filename"] for r in results_data]
            )

            # Display selected result
            selected_result = next(r for r in results_data if r["filename"] == result_choice)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Model", selected_result.get("model", "Unknown"))
            with col2:
                st.metric("Total Problems", selected_result.get("num_problems", "N/A"))
            with col3:
                st.metric("Samples per Problem", selected_result.get("num_samples_per_problem", "N/A"))

            # Pass@k metrics
            if "pass_at_k" in selected_result:
                st.subheader("Performance Metrics")
                pass_at_k = selected_result["pass_at_k"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pass@1", f"{pass_at_k.get('pass@1', 0)*100:.1f}%")
                with col2:
                    st.metric("Pass@5", f"{pass_at_k.get('pass@5', 0)*100:.1f}%")
                with col3:
                    st.metric("Pass@10", f"{pass_at_k.get('pass@10', 0)*100:.1f}%")

            # Energy metrics
            if "gpu_metrics" in selected_result:
                st.subheader("Energy Consumption")
                gpu_metrics = selected_result["gpu_metrics"]

                # Calculate total energy
                total_energy = sum(
                    v for k, v in gpu_metrics.items()
                    if k.endswith("_energy_wh")
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Energy", f"{total_energy:.2f} Wh")
                with col2:
                    st.metric("Generation Time", f"{selected_result.get('generation_time_sec', 0)/60:.1f} min")

            # Full JSON
            with st.expander("View Full JSON"):
                st.json(selected_result)

        else:
            st.warning("No evaluation results found in results/ directory")

    except Exception as e:
        st.error(f"Error loading results: {e}")


# Tab 4: About
with tab4:
    st.header("About This Dashboard")

    st.markdown("""
    ## LLM Compression Evaluation Dashboard

    This interactive dashboard allows you to:

    ### 🎯 Test Models
    - **Student (8B Base):** Baseline Llama-3.1-8B model
    - **Distilled (8B + LoRA):** Knowledge-distilled model with LoRA adapters
    - **Teacher (70B):** Large Llama-3.1-70B model (8-bit quantized)

    ### 📊 Evaluate Performance
    - Generate solutions for HumanEval coding problems
    - Compare models side-by-side
    - View pass@k metrics and energy consumption

    ### 🔬 Track Experiments
    - Log all generations to MLflow
    - Review past evaluation results
    - Monitor GPU usage and costs

    ---

    ### 📁 Resource Locations

    **Models (cached):**
    - System cache: `~/.cache/huggingface/hub/`
    - Teacher (70B): 132 GB
    - Student (8B): 15 GB

    **LoRA Adapters:**
    - `outputs/llama31_8b_kd_lora/lora/` (161 MB)

    **HumanEval Dataset:**
    - Bundled with `human-eval` package (164 problems)

    **MLflow Tracking:**
    - `mlruns/` directory

    ---

    ### 🚀 Quick Start

    1. Select a model from the sidebar
    2. Choose a HumanEval problem
    3. Adjust generation parameters
    4. Click "Generate Solution"
    5. Review outputs and MLflow logs

    ---

    Built with Streamlit • Powered by Llama 3.1
    """)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**MLflow UI:**
Start with: `mlflow ui --port 5000`
View at: http://localhost:5000
""")
