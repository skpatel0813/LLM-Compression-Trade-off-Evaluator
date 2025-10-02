# scripts/run_kd_mlflow.py
import os, time, json, subprocess, sys, pathlib
import mlflow

def main():
    # Point MLflow to a local folder (easy mode)
    tracking_dir = os.environ.get("MLFLOW_TRACKING_DIR", "mlruns")
    pathlib.Path(tracking_dir).mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{tracking_dir}"

    exp_name = os.environ.get("MLFLOW_EXPERIMENT", "llama31_kd")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=os.environ.get("RUN_NAME", "kd_lora")):
        # Log key env vars & config yaml as artifact
        mlflow.log_params({
            "MAX_SAMPLES": os.environ.get("MAX_SAMPLES", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES",""),
            "bf16": os.environ.get("BF16",""),
            "teacher_4bit": os.environ.get("TEACHER_4BIT",""),
            "teacher_8bit": os.environ.get("TEACHER_8BIT",""),
        })
        cfg_path = "configs/project.yaml"
        if os.path.isfile(cfg_path):
            mlflow.log_artifact(cfg_path, artifact_path="configs")

        # Run training as a module so your code stays unchanged
        cmd = [sys.executable, "-u", "-m", "src.train_kd"]
        print(">> running:", " ".join(cmd))
        t0 = time.time()
        rc = subprocess.call(cmd)
        dur = time.time() - t0
        mlflow.log_metric("train_wall_seconds", dur)

        # Log LoRA folder as artifact if present
        lora_dir = "outputs/lora/llama31_8b_kd_lora"
        out_dir  = "outputs/llama31_8b_kd_lora"
        if os.path.isdir(lora_dir):
            mlflow.log_artifacts(lora_dir, artifact_path="lora")
        if os.path.isdir(out_dir):
            mlflow.log_artifacts(out_dir, artifact_path="trainer_outputs")

        sys.exit(rc)

if __name__ == "__main__":
    main()
