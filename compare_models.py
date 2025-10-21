import json
from pathlib import Path

def load_metrics(run_name, cleaned=False):
    """Load metrics from a run directory."""
    suffix = "_clean" if cleaned else ""
    metrics_file = f"outputs/eval/{run_name}/metrics_eval.json"
    
    if Path(metrics_file).exists():
        return json.load(open(metrics_file))
    return None

def compute_cleaned_metrics(run_name):
    """Compute metrics on cleaned predictions."""
    import sys
    sys.path.insert(0, '.')
    from src.codebleu_compat import calc_codebleu
    
    clean_file = f"outputs/eval/{run_name}/predictions_clean.jsonl"
    if not Path(clean_file).exists():
        return None
    
    refs, hyps = [], []
    with open(clean_file) as f:
        for line in f:
            rec = json.loads(line)
            ref = rec.get('reference', '').strip()
            hyp = rec.get('prediction', '').strip()
            if ref and hyp:
                refs.append(ref)
                hyps.append(hyp)
    
    if not refs:
        return None
    
    result = calc_codebleu(refs, hyps, lang='python')
    return {
        'codebleu': result['codebleu'],
        'ngram': result['ngram_match'],
        'syntax': result['syntax_match'],
        'dataflow': result['dataflow_match']
    }

print("="*80)
print("MODEL COMPARISON REPORT")
print("="*80)

# Student
student_raw = load_metrics('student-8b-both-test')
student_clean = compute_cleaned_metrics('student-8b-both-test')

# Teacher
teacher_raw = load_metrics('teacher-70b-baseline')
teacher_clean = compute_cleaned_metrics('teacher-70b-baseline')

print("\n1. RAW PREDICTIONS (with system prompts)")
print("-"*80)
print(f"{'Metric':<20} {'8B Student':<15} {'70B Teacher':<15} {'Difference':<15}")
print("-"*80)

if student_raw and teacher_raw:
    for metric in ['bleu4', 'codebleu', 'codebleu_syntax', 'codebleu_dataflow']:
        s = student_raw.get(metric, 0)
        t = teacher_raw.get(metric, 0)
        diff = t - s
        print(f"{metric:<20} {s:<15.4f} {t:<15.4f} {diff:+.4f}")

print("\n2. CLEANED PREDICTIONS (code only)")
print("-"*80)
print(f"{'Metric':<20} {'8B Student':<15} {'70B Teacher':<15} {'Difference':<15}")
print("-"*80)

if student_clean and teacher_clean:
    for metric in ['codebleu', 'ngram', 'syntax', 'dataflow']:
        s = student_clean.get(metric, 0)
        t = teacher_clean.get(metric, 0)
        diff = t - s
        sign = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "≈")
        print(f"{metric:<20} {s:<15.4f} {t:<15.4f} {diff:+.4f} {sign}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if student_clean and teacher_clean:
    s_cb = student_clean['codebleu']
    t_cb = teacher_clean['codebleu']
    improvement = ((t_cb - s_cb) / s_cb) * 100 if s_cb > 0 else 0
    
    print(f"Teacher improvement over Student: {improvement:+.1f}%")
    print(f"\nKey takeaway:")
    if improvement > 15:
        print("  ✓ Teacher significantly outperforms student")
        print("  ✓ Knowledge distillation should provide substantial gains")
    elif improvement > 5:
        print("  ≈ Teacher moderately outperforms student")
        print("  ≈ Knowledge distillation may provide moderate gains")
    else:
        print("  ! Teacher and student perform similarly")
        print("  ! This suggests:")
        print("    - Both models generate creative alternatives")
        print("    - Dataset may be too easy/ambiguous")
        print("    - Consider using HumanEval or MBPP for clearer signal")

print("="*80)
