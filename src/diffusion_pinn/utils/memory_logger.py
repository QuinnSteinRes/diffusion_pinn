import os
import psutil
import threading
import time
from datetime import datetime

__all__ = ['MemoryMonitor']

class MemoryMonitor:
    def __init__(self, log_file='memory_usage.log'):
        self.log_file = log_file
        self.is_running = False
        self.monitor_thread = None

    def _log_memory(self):
        process = psutil.Process(os.getpid())
        while self.is_running:
            try:
                mem_info = process.memory_info()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp}, RSS: {mem_info.rss / 1024 / 1024:.2f}MB, "
                           f"VMS: {mem_info.vms / 1024 / 1024:.2f}MB\n")
                time.sleep(10)  # Log every 10 seconds
            except Exception as e:
                print(f"Memory monitoring error: {str(e)}")
                break

    def start(self):
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._log_memory)
        self.monitor_thread.start()

    def stop(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
