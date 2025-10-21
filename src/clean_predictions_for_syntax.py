#!/usr/bin/env python3
"""
Clean predictions to extract only code, removing markdown fences and preamble text.
"""
import re
import sys
import json
import ast

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text: str) -> str:
    """Extract code from text, removing markdown and preamble."""
    if not text:
        return ""
    
    # Try to find code in markdown fences
    m = FENCE_RE.search(text)
    code = m.group(1) if m else text
    
    # Strip leftover fences
    code = code.replace("```", "")
    
    # Drop common preamble lines
    lines = []
    skip_phrases = [
        "here is", "here's", "the function", "explanation", 
        "system", "cutting knowledge date", "today date",
        "you are a helpful", "i can help"
    ]
    
    for ln in code.splitlines():
        stripped = ln.strip().lower()
        # Skip empty lines and common preamble
        if not stripped:
            continue
        if any(phrase in stripped for phrase in skip_phrases):
            continue
        lines.append(ln)
    
    code = "\n".join(lines).strip()
    
    # Ensure final newline
    if code and not code.endswith("\n"):
        code += "\n"
    
    return code

def is_parseable_py(code: str) -> bool:
    """Check if code is valid Python."""
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def main(input_file: str, output_file: str):
    """Clean predictions from input file and write to output file."""
    total, parseable = 0, 0
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in fin:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract fields (handle different naming conventions)
            pred = rec.get("hyp") or rec.get("prediction") or rec.get("output") or rec.get("generated") or ""
            ref = rec.get("ref") or rec.get("reference") or rec.get("target") or rec.get("gold") or ""
            
            # Clean both prediction and reference
            pred_clean = extract_code(pred)
            ref_clean = extract_code(ref) if ref else ref
            
            # Check if prediction is valid Python
            if pred_clean and is_parseable_py(pred_clean):
                parseable += 1
            
            # Write cleaned record
            out_rec = {
                "prediction": pred_clean,
                "reference": ref_clean,
                "prompt": rec.get("prompt", ""),
                "original_prediction": pred,
                "original_reference": ref
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
    
    print(f"[cleaner] Processed: {total} examples")
    print(f"[cleaner] Parseable predictions: {parseable} ({parseable/total*100:.1f}%)")
    print(f"[cleaner] Output: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_predictions_for_syntax.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
