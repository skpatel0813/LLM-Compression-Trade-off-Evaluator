#!/usr/bin/env python
import re, sys, json, ast

FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text: str) -> str:
    if not text:
        return ""
    m = FENCE_RE.search(text)
    code = m.group(1) if m else text  # if no fence, take raw text
    # Strip leftover fences and chatter lines
    code = code.replace("```", "")
    # Drop common preambles
    lines = [ln for ln in code.splitlines()
             if not ln.strip().lower().startswith(("here is", "here's", "the function", "explanation"))]
    code = "\n".join(lines).strip()
    # Ensure final newline
    if not code.endswith("\n"):
        code += "\n"
    return code

def is_parseable_py(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def main(inp, outp):
    total, ok = 0, 0
    with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            rec = json.loads(line)
            # your file likely has these fields; adjust if named differently:
            pred = rec.get("prediction") or rec.get("output") or rec.get("generated") or ""
            ref  = rec.get("reference") or rec.get("target") or rec.get("gold") or ""
            pred_clean = extract_code(pred)
            # If unparseable, try a simple fallback: close fences already removed; last line newline ensured above.
            if is_parseable_py(pred_clean):
                ok += 1
            # Write a new record with cleaned fields
            rec["prediction"] = pred_clean
            rec["reference"]  = extract_code(ref) if ref else ref
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[cleaner] cleaned={total}, parseable_pred={ok} ({ok/total:.1%})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: clean_predictions_for_syntax.py <in.jsonl> <out.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
