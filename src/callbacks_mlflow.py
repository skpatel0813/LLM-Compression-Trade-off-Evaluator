# src/callbacks_mlflow.py
# =================================================================================================
# A lightweight MLflow callback for Hugging Face Trainer that:
#   • Starts/finishes an MLflow run automatically
#   • Logs Trainer hyperparams (batch size, lr, precision, etc.)
#   • Logs training metrics at the same cadence as Trainer logging
#   • Logs GPU telemetry each N steps (utilization, memory, temperature, power)
#     – via pynvml if available (no-op if not installed)
#   • Logs checkpoint paths when Trainer saves
#
# Safe by default:
#   • If MLflow is not installed or a tracking URI is not configured, the callback
#     will simply print a short message and become a no-op (training continues).
#   • If pynvml isn’t installed or NVML is unavailable, GPU telemetry is skipped.
#
# Environment variables you can use:
#   • MLFLOW_TRACKING_URI        – e.g., http://mlflow.your.domain:5000  (or a local path)
#   • MLFLOW_EXPERIMENT_NAME     – experiment to log under (defaults: "LLM-KD")
#   • MLFLOW_RUN_NAME            – name of the run (defaults derived from models + LoRA on/off)
#
# Typical use:
#   from callbacks_mlflow import LiteMLflowCallback
#   trainer = Trainer(..., callbacks=[LiteMLflowCallback(log_gpu_every_n_steps=20)])
#
# =================================================================================================

from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, List

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy

# Optional imports – these are *not* hard requirements
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_NVML = False


def _safe_get_visible_gpus() -> List[int]:
    """Return list of visible CUDA device indices, or [] if CUDA not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _nvml_init_once() -> bool:
    """Initialize NVML once; return True if usable."""
    if not _HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _collect_gpu_snapshot() -> Dict[str, float]:
    """
    Collect a one-shot snapshot of GPU metrics across all visible devices.
    Returns a flat dict of aggregated stats:
      gpu_count, util_mean, mem_used_GiB, mem_total_GiB, mem_frac, power_W, temp_C
    If NVML unavailable, returns empty dict.
    """
    if not _HAS_NVML:
        return {}

    try:
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return {}

    if count == 0:
        return {}

    util_vals = []
    mem_used = []
    mem_total = []
    power_vals = []
    temp_vals = []

    for i in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_vals.append(float(util.gpu))

            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used.append(float(mem.used) / (1024 ** 3))
            mem_total.append(float(mem.total) / (1024 ** 3))

            # Power (mW -> W); some systems return 0 if power readings disabled
            p = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_vals.append(float(p) / 1000.0)

            # Temperature (C)
            t = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            temp_vals.append(float(t))
        except Exception:
            # On any device failure, skip that device and continue
            continue

    if not util_vals:
        return {}

    def _mean(xs: List[float]) -> float:
        return sum(xs) / max(1, len(xs))

    total_mem = sum(mem_total)
    used_mem = sum(mem_used)
    mem_frac = used_mem / total_mem if total_mem > 0 else 0.0

    return {
        "gpu_count": float(len(util_vals)),
        "gpu_util_mean_pct": _mean(util_vals),
        "gpu_mem_used_GiB": used_mem,
        "gpu_mem_total_GiB": total_mem,
        "gpu_mem_frac": mem_frac,
        "gpu_power_W_mean": _mean(power_vals) if power_vals else 0.0,
        "gpu_temp_C_mean": _mean(temp_vals) if temp_vals else 0.0,
    }


class LiteMLflowCallback(TrainerCallback):
    """
    Minimal, robust MLflow callback.

    Args:
        log_gpu_every_n_steps: int
            How often to log GPU telemetry (util/mem/power/temp). If None or <=0, disables GPU logging.
        extra_tags: Optional[Dict[str, str]]
            Extra MLflow tags to attach to the run.
        run_name: Optional[str]
            Force a run name (otherwise auto / env).
        experiment_name: Optional[str]
            Force an experiment (otherwise env: MLFLOW_EXPERIMENT_NAME, fallback "LLM-KD").
    """

    def __init__(
        self,
        log_gpu_every_n_steps: Optional[int] = 20,
        extra_tags: Optional[Dict[str, str]] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        super().__init__()
        self.log_gpu_every_n_steps = (log_gpu_every_n_steps or 0)
        self.extra_tags = extra_tags or {}
        self.run_name = run_name
        self.experiment_name = experiment_name
        self._mlflow_active = False
        self._nvml_ready = False
        self._last_gpu_log_step = -1

    # ------------- helpers

    def _start_mlflow_if_possible(self, args, model) -> None:
        if not _HAS_MLFLOW:
            print("[LiteMLflowCallback] MLflow not installed; telemetry disabled.")
            return

        # Tracking URI (optional). If not set, MLflow defaults to local ./mlruns
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            try:
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as e:
                print(f"[LiteMLflowCallback] Failed to set tracking URI: {e}")

        # Experiment name
        exp_name = self.experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or "LLM-KD"
        try:
            mlflow.set_experiment(exp_name)
        except Exception as e:
            print(f"[LiteMLflowCallback] Failed to set experiment '{exp_name}': {e}")

        # Run name
        if self.run_name:
            run_name = self.run_name
        else:
            # derive a helpful default
            student_name = getattr(getattr(model, "config", None), "name_or_path", None) or "student"
            run_name = os.environ.get("MLFLOW_RUN_NAME") or f"KD-{student_name}"

        try:
            mlflow.start_run(run_name=run_name)
            self._mlflow_active = True
        except Exception as e:
            print(f"[LiteMLflowCallback] Failed to start MLflow run: {e}")
            self._mlflow_active = False
            return

        # Attach tags
        try:
            tags = {
                "trainer": "hf_trainer",
                "strategy": "knowledge_distillation",
                "interval_strategy": str(args.logging_strategy),
            }
            tags.update(self.extra_tags)
            mlflow.set_tags(tags)
        except Exception as e:
            print(f"[LiteMLflowCallback] Failed to set MLflow tags: {e}")

        # Log static params
        try:
            params = {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "warmup_ratio": args.warmup_ratio,
                "seq_len": getattr(args, "max_seq_length", None) or "from_collator",
                "bf16": getattr(args, "bf16", False),
                "fp16": getattr(args, "fp16", False),
                "optim": args.optim,
                "lr_scheduler_type": str(args.lr_scheduler_type),
                "weight_decay": args.weight_decay,
                "dataloader_num_workers": args.dataloader_num_workers,
                "remove_unused_columns": args.remove_unused_columns,
            }
            # Visible GPUs info
            gpus = _safe_get_visible_gpus()
            params["visible_gpus"] = ",".join(map(str, gpus)) if gpus else "CPU"
            mlflow.log_params(params)
        except Exception as e:
            print(f"[LiteMLflowCallback] Failed to log MLflow params: {e}")

    def _end_mlflow_if_active(self) -> None:
        if _HAS_MLFLOW and self._mlflow_active:
            try:
                mlflow.end_run()
            except Exception as e:
                print(f"[LiteMLflowCallback] Failed to end MLflow run: {e}")
            self._mlflow_active = False

    # ------------- TrainerCallback hooks

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Start MLflow run
        self._start_mlflow_if_possible(args, kwargs.get("model"))
        # Initialize NVML if we plan to log GPU telemetry
        if self.log_gpu_every_n_steps > 0:
            self._nvml_ready = _nvml_init_once()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """
        Trainer calls this at its logging cadence.
        We mirror those logs to MLflow and also add GPU telemetry (every N steps).
        """
        step = int(state.global_step or 0)

        # Log trainer-provided metrics
        if _HAS_MLFLOW and self._mlflow_active:
            try:
                mlflow.log_metrics(logs, step=step)
            except Exception as e:
                print(f"[LiteMLflowCallback] Failed to log metrics to MLflow: {e}")

        # Possibly add GPU telemetry
        if self.log_gpu_every_n_steps > 0 and self._nvml_ready:
            if (self._last_gpu_log_step < 0) or (step - self._last_gpu_log_step >= self.log_gpu_every_n_steps):
                snap = _collect_gpu_snapshot()
                if snap and _HAS_MLFLOW and self._mlflow_active:
                    try:
                        mlflow.log_metrics(snap, step=step)
                    except Exception as e:
                        print(f"[LiteMLflowCallback] Failed to log GPU snapshot: {e}")
                self._last_gpu_log_step = step

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log checkpoint directory path when Trainer saves."""
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if _HAS_MLFLOW and self._mlflow_active:
            try:
                mlflow.log_params({"last_checkpoint_dir": ckpt_dir})
            except Exception as e:
                print(f"[LiteMLflowCallback] Failed to log checkpoint path: {e}")

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Finish MLflow run."""
        self._end_mlflow_if_active()
        # Try to shutdown NVML cleanly
        if self._nvml_ready:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
