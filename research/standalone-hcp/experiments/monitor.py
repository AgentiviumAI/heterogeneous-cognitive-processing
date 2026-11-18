import threading
import time

class Monitor:
    def __init__(self):
        self.nv = None
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            parent = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(parent, 0) if pynvml.nvmlDeviceGetMigMode(parent)[0] else parent
            self.nv = pynvml
        except Exception as exception:
            self.unavailable_reason = f'{type(exception).__name__}: {exception}'
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        if self.nv is None:
            return
        while not self.stop_event.is_set():
            try:
                memory = self.nv.nvmlDeviceGetMemoryInfo(self.handle)
            except self.nv.NVMLError:
                self.stop_event.wait(0.5)
                continue
            try:
                utilization = self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu
            except self.nv.NVMLError:
                utilization = None
            self.rows.append({'time': time.time(), 'used_bytes': memory.used, 'total_bytes': memory.total, 'gpu_utilization_percent': utilization})
            self.stop_event.wait(0.5)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        return self.rows
