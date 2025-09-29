"""
Download & format Python-only training data into chat JSONL files.

Outputs:
  data/opencodeinstruct_python_train.jsonl
  data/codesearchnet_python_train.jsonl
"""

import os, json, random
from datasets import load_dataset
from .utils import Config

def to_chat(instruction: str, _input: str, output: str) -> dict:
    system = (
        "You are a precise Python coding assistant. "
        "Prefer minimal, runnable code and include brief tests when sensible."
    )
    user = (instruction or "")
    if _input: user += ("\n" + _input)
    return {"messages":[
        {"role":"system","content":system},
        {"role":"user","content":user},
        {"role":"assistant","content":output or ""}
    ]}

def main():
    cfg = Config.load().cfg
    outdir = cfg["paths"]["data_dir"]
    os.makedirs(outdir, exist_ok=True)
    random.seed(1234)

    # 1) OpenCodeInstruct (Python subset)
    print("Downloading nvidia/OpenCodeInstruct…")
    oci = load_dataset("nvidia/OpenCodeInstruct", split="train")
    def is_py(x): 
        lang = (x.get("language") or "").lower()
        return "python" in lang or "py" in lang
    oci_py = oci.filter(is_py).shuffle(seed=1234)
    with open(os.path.join(outdir, "opencodeinstruct_python_train.jsonl"), "w", encoding="utf-8") as f:
        for ex in oci_py:
            f.write(json.dumps(to_chat(ex.get("instruction"), ex.get("input"), ex.get("output"))) + "\n")

    # 2) CodeSearchNet (Python)
    print("Downloading CodeSearchNet (python)…")
    csn = load_dataset("code_search_net", "python")
    pairs = []
    for split in ["train","validation"]:
        for ex in csn[split]:
            doc = (ex.get("func_documentation_string") or "").strip()
            code= (ex.get("func_code_string") or "").strip()
            if doc and code:
                pairs.append(to_chat("Write the Python function described by this docstring:", doc, code))
    random.shuffle(pairs)
    with open(os.path.join(outdir, "codesearchnet_python_train.jsonl"), "w", encoding="utf-8") as f:
        for rec in pairs: f.write(json.dumps(rec) + "\n")
    print("✅ Data prepared.")

if __name__ == "__main__":
    main()
