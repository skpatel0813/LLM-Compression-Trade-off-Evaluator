# src/gpu_monitor.py
import csv
import os
import threading
import time
from datetime import datetime

try:
    try:
        import pynvml as nvml  # deprecated name but present in many envs
    except Exception:
        import nvidia_ml_py3 as nvml  # alternative package name
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


class GPUMonitor:
    """
    Polls NVML every `interval_sec` and writes:
    timestamp,gpu_index,utilization_pct,mem_used_mb,mem_total_mb,power_w
    """
    def __init__(self, out_csv: str, interval_sec: float = 1.0):
        self.out_csv = out_csv
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thr = None

        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if _HAS_NVML:
            try:
                nvml.nvmlInit()
            except Exception:
                pass

    def start(self):
        if not _HAS_NVML:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        if not _HAS_NVML:
            return
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass

    def _loop(self):
        header_written = os.path.isfile(self.out_csv) and os.path.getsize(self.out_csv) > 0
        with open(self.out_csv, "a", newline="") as f:
            wr = csv.writer(f)
            if not header_written:
                wr.writerow(["timestamp", "gpu_index", "utilization_pct", "mem_used_mb", "mem_total_mb", "power_w"])
            while not self._stop.is_set():
                try:
                    n = nvml.nvmlDeviceGetCount()
                    ts = datetime.utcnow().isoformat()
                    for i in range(n):
                        h = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(h)
                        mem = nvml.nvmlDeviceGetMemoryInfo(h)
                        power_mw = 0
                        try:
                            power_mw = nvml.nvmlDeviceGetPowerUsage(h)  # in milliwatts
                        except Exception:
                            pass
                        wr.writerow([
                            ts,
                            i,
                            getattr(util, "gpu", 0),
                            int(mem.used / (1024 * 1024)),
                            int(mem.total / (1024 * 1024)),
                            round(power_mw / 1000.0, 2),
                        ])
                    f.flush()
                except Exception:
                    # Keep sampling even if a read fails
                    pass
                time.sleep(self.interval_sec)
