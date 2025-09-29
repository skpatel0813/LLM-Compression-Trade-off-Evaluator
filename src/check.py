# Quick environment smoke test: HF auth (gated repo access) + CUDA + Git LFS.
import os, subprocess, shutil
from huggingface_hub import whoami, snapshot_download

def print_kv(k, v): print(f"{k}: {v}")

print("=== Hugging Face ===")
try:
    me = whoami()
    print_kv("User", me.get("name"))
    # Tiny gated-file probe (no big downloads)
    snapshot_download("meta-llama/Meta-Llama-3.1-8B-Instruct",
                      allow_patterns=["config.json"])
    print("HF gated access: OK")
except Exception as e:
    print("HF check FAILED ->", e)

print("\n=== CUDA / PyTorch ===")
try:
    import torch
    print_kv("CUDA available", torch.cuda.is_available())
    print_kv("GPU count", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print_kv(f"GPU{i}", torch.cuda.get_device_name(i))
except Exception as e:
    print("PyTorch/CUDA check FAILED ->", e)

print("\n=== Git LFS ===")
lfs = shutil.which("git")
if lfs:
    try:
        out = subprocess.check_output(["git", "lfs", "version"], text=True).strip()
        print_kv("git lfs", out)
    except Exception as e:
        print("Git LFS check FAILED ->", e)
else:
    print("Git not found on PATH")

print("\n=== Writable dirs ===")
for p in ["data", "outputs", "reports"]:
    try:
        os.makedirs(p, exist_ok=True)
        testfile = os.path.join(p, ".write_test")
        open(testfile, "w").write("ok")
        os.remove(testfile)
        print_kv(p, "writable")
    except Exception as e:
        print_kv(p, f"NOT writable -> {e}")

print("\nDone.")
