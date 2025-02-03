import os
import time
import psutil
import threading
from datetime import datetime

class MemoryLogger:
    def __init__(self, output_dir, log_interval=30):
        """
        Initialize memory logger
        
        Args:
            output_dir: Directory to save memory log file
            log_interval: Interval in seconds between memory measurements
        """
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.is_running = False
        self.start_time = None
        self.log_file = os.path.join(output_dir, 'memory_usage.log')
        
    def _get_memory_usage(self):
        """Get current memory usage stats"""
        process = psutil.Process(os.getpid())
        
        # Get memory info
        mem_info = process.memory_info()
        
        # Get system memory info
        sys_mem = psutil.virtual_memory()
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rss': mem_info.rss / (1024 * 1024),  # RSS in MB
            'vms': mem_info.vms / (1024 * 1024),  # VMS in MB
            'percent': process.memory_percent(),
            'sys_total': sys_mem.total / (1024 * 1024),  # Total system memory in MB
            'sys_available': sys_mem.available / (1024 * 1024),  # Available system memory in MB
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
        
    def _log_header(self):
        """Write header to log file"""
        with open(self.log_file, 'w') as f:
            f.write("timestamp,elapsed_time,rss_mb,vms_mb,memory_percent,sys_total_mb,sys_available_mb\n")
            
    def _log_memory(self):
        """Log memory usage periodically"""
        while self.is_running:
            mem_stats = self._get_memory_usage()
            
            with open(self.log_file, 'a') as f:
                f.write(f"{mem_stats['timestamp']},{mem_stats['elapsed_time']:.2f},"
                       f"{mem_stats['rss']:.2f},{mem_stats['vms']:.2f},{mem_stats['percent']:.2f},"
                       f"{mem_stats['sys_total']:.2f},{mem_stats['sys_available']:.2f}\n")
                
            time.sleep(self.log_interval)
            
    def start(self):
        """Start memory logging"""
        self.is_running = True
        self.start_time = time.time()
        self._log_header()
        
        # Start logging in a separate thread
        self.thread = threading.Thread(target=self._log_memory)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop memory logging"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def plot_memory_usage(self, save_path=None):
        """Plot memory usage over time"""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            # Read log file
            df = pd.read_csv(self.log_file)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot memory usage
            ax1.plot(df['elapsed_time']/3600, df['rss_mb']/1024, label='RSS')
            ax1.plot(df['elapsed_time']/3600, df['vms_mb']/1024, label='VMS')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Memory Usage (GB)')
            ax1.grid(True)
            ax1.legend()
            
            # Plot system memory
            ax2.plot(df['elapsed_time']/3600, df['sys_available_mb']/1024, label='Available')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('System Memory (GB)')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting memory usage: {str(e)}")
