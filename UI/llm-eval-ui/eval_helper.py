"""
Helper functions for HumanEval evaluation.

This module handles the quirks of the HumanEval evaluation framework,
particularly the requirement that ALL 164 problems must be in the JSONL file.
"""
import json
import tempfile
from pathlib import Path
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness


def test_single_problem(problem_id: str, completions: list) -> dict:
    """
    Test solutions for a single HumanEval problem.

    Args:
        problem_id: The HumanEval problem ID (e.g., "HumanEval/0")
        completions: List of completion strings (function bodies with proper indentation)

    Returns:
        dict with:
            - success: bool
            - num_correct: int (number of solutions that passed)
            - num_total: int (total solutions tested)
            - detailed_results: list of per-solution results
            - error: str (if failed)

    IMPORTANT: The HumanEval evaluation framework requires ALL 164 problems
    to be present in the JSONL file. We work around this by:
    1. Adding placeholder empty completions for all other problems
    2. Only testing the solutions for the target problem
    3. Extracting only the results for the target problem
    """
    # Load all problems (needed for placeholders)
    all_problems = read_problems()

    if problem_id not in all_problems:
        return {
            'success': False,
            'error': f'Problem {problem_id} not found in HumanEval dataset',
            'num_correct': 0,
            'num_total': len(completions)
        }

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

        # Write ONE entry per problem ID (evaluate_functional_correctness expects unique task_ids)
        # For the target problem: write multiple copies with a counter approach
        # Actually, we can only have ONE entry per task_id, so we'll test each completion separately

        # Strategy: Write all problems with empty completions, EXCEPT our target
        # For our target, we'll test completions one at a time
        pass

    # Better strategy: Test each completion separately and aggregate results
    results = []
    for i, completion in enumerate(completions):
        result = _test_single_completion(problem_id, completion, all_problems)
        results.append(result)

    # Aggregate results
    num_correct = sum(1 for r in results if r.get('passed', False))
    num_total = len(completions)

    return {
        'success': True,
        'num_correct': num_correct,
        'num_total': num_total,
        'detailed_results': results
    }


def _test_single_completion(problem_id: str, completion: str, all_problems: dict) -> dict:
    """
    Test a single completion for a single problem.

    Returns:
        dict with 'passed' (bool), 'result' (str with error if failed)
    """
    # Create temp file with placeholder entries for all problems
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

        # Write entries for ALL 164 problems
        for pid in sorted(all_problems.keys()):
            if pid == problem_id:
                # The problem we're testing
                entry = {
                    "task_id": pid,
                    "completion": completion
                }
            else:
                # Placeholder - empty completion (will fail, but we don't care)
                entry = {
                    "task_id": pid,
                    "completion": ""
                }
            f.write(json.dumps(entry) + '\n')

    try:
        # Run evaluation
        # This will evaluate all 164 problems, but we only care about our target
        evaluate_functional_correctness(temp_file, k=[1], n_workers=4, timeout=3.0)

        # Read detailed results
        results_file = temp_file + "_results.jsonl"
        if not Path(results_file).exists():
            return {
                'passed': False,
                'result': 'Results file not found'
            }

        # Find the result for our target problem
        with open(results_file) as f:
            for line in f:
                result = json.loads(line)
                if result['task_id'] == problem_id:
                    # Clean up temp files
                    Path(temp_file).unlink()
                    Path(results_file).unlink()

                    return {
                        'passed': result.get('passed', False),
                        'result': result.get('result', '')
                    }

        # Didn't find our problem in results
        Path(temp_file).unlink()
        if Path(results_file).exists():
            Path(results_file).unlink()

        return {
            'passed': False,
            'result': f'Problem {problem_id} not found in results'
        }

    except Exception as e:
        # Clean up on error
        if Path(temp_file).exists():
            Path(temp_file).unlink()
        if Path(temp_file + "_results.jsonl").exists():
            Path(temp_file + "_results.jsonl").unlink()

        return {
            'passed': False,
            'result': f'Evaluation error: {str(e)}'
        }
