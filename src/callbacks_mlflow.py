# src/callbacks_mlflow.py
import os, time, json, platform
from dataclasses import asdict
from typing import Optional
import torch
from transformers.integrations import MLflowCallback
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

def _gpu_snapshot():
    if not torch.cuda.is_available() or not _HAS_NVML:
        return {}
    try:
        pynvml.nvmlInit()
        gpus = []
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            pwr = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpus.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(h).decode(),
                "mem_used_gb": round(mem.used/1024**3, 3),
                "mem_total_gb": round(mem.total/1024**3, 3),
                "power_w": round(pwr/1000.0, 2),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
            })
        return {"gpus": gpus}
    except Exception:
        return {}
    finally:
        try: pynvml.nvmlShutdown()
        except: pass

class LiteMLflowCallback(MLflowCallback):
    """
    - Logs trainer logs (loss/learning rate/step) via parent class
    - Also logs periodic GPU snapshots & environment info
    """
    def __init__(self, log_gpu_every_n_steps: int = 20):
        super().__init__()
        self._n = int(log_gpu_every_n_steps)
        self._last = 0

    def setup(self, args, state, model):
        super().setup(args, state, model)
        # static env info once
        self._ml_flow.log_params({
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch": torch.__version__,
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        })

    def on_log(self, args, state, control, **kwargs):
        # parent logs loss/lr/etc
        super().on_log(args, state, control, **kwargs)
        if state.global_step - self._last >= self._n:
            self._last = state.global_step
            snap = _gpu_snapshot()
            if snap:
                # log as metrics (first GPU) + as artifact (full JSON)
                g0 = snap["gpus"][0]
                self._ml_flow.log_metrics({
                    "gpu0_mem_used_gb": g0["mem_used_gb"],
                    "gpu0_util_pct": g0["gpu_util_pct"],
                    "gpu0_power_w": g0["power_w"],
                }, step=state.global_step)
