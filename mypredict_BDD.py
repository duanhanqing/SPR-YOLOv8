# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO(r'/szzn/luoluo/work2/YOLOv8.2/消融实验/BDD_100K_300/7YOLOv8_SGBackbone_SlimNeck_DetectSA/train/exp/weights/best.pt') # select your model.pt path
#     model.predict(source=r'/szzn/luoluo/work2/YOLOv8.2/datasets/BDD_100K/images/test',
#                   # split='test',  # 使用测试集
#                   imgsz=640,
#                   name='exp',
#                   save=True,
#                   save_conf = True,
#                   batch=64,
#                   workers=8,
#                   device='7',
#                   # classes=0,
#                   project= '/szzn/luoluo/work2/YOLOv8.2/消融实验/BDD_100K_300/7YOLOv8_SGBackbone_SlimNeck_DetectSA/test',
#                 )

import warnings
import time
import psutil
import torch
from threading import Thread, Event
from collections import deque
from ultralytics import YOLO

warnings.filterwarnings('ignore')

class SystemMonitor:
    """系统资源监控类"""
    def __init__(self):
        self.monitor_event = Event()
        self.cpu_usage = deque(maxlen=100)  # 记录CPU使用率
        self.mem_usage = deque(maxlen=100)  # 记录内存使用量(GB)
        self.gpu_usage = deque(maxlen=100)  # 记录GPU使用率
        self.gpu_mem = deque(maxlen=100)    # 记录GPU显存使用量(GB)

    def start(self):
        """启动监控线程"""
        self.monitor_event.set()
        Thread(target=self._monitor_loop).start()

    def stop(self):
        """停止监控"""
        self.monitor_event.clear()

    def _monitor_loop(self):
        """监控循环"""
        while self.monitor_event.is_set():
            # 记录CPU使用率
            self.cpu_usage.append(psutil.cpu_percent(interval=0.5))
            
            # 记录内存使用
            mem = psutil.virtual_memory()
            self.mem_usage.append(mem.used / (1024 ** 3))  # 转换为GB
            
            # 记录GPU使用（如果可用）
            if torch.cuda.is_available():
                self.gpu_usage.append(torch.cuda.utilization(0))
                self.gpu_mem.append(torch.cuda.memory_allocated(0) / (1024 ** 3))
            
            time.sleep(0.5)  # 采样间隔

def print_report(monitor, total_time):
    """打印资源使用报告"""
    print("\n" + "="*40)
    print("硬件资源使用报告:")
    print(f"总运行时间: {total_time:.2f}秒")
    
    # CPU信息
    print(f"\nCPU平均使用率: {sum(monitor.cpu_usage)/len(monitor.cpu_usage):.1f}%")
    print(f"CPU峰值使用率: {max(monitor.cpu_usage)}%")
    
    # 内存信息
    print(f"\n内存平均使用: {sum(monitor.mem_usage)/len(monitor.mem_usage):.1f}GB")
    print(f"内存峰值使用: {max(monitor.mem_usage):.1f}GB")
    
    # GPU信息（如果可用）
    if torch.cuda.is_available() and len(monitor.gpu_usage) > 0:
        print(f"\nGPU平均使用率: {sum(monitor.gpu_usage)/len(monitor.gpu_usage):.1f}%")
        print(f"GPU峰值使用率: {max(monitor.gpu_usage)}%")
        print(f"GPU显存峰值: {max(monitor.gpu_mem):.1f}GB")
    print("="*40 + "\n")

if __name__ == '__main__':
    try:
        # 初始化监控器
        monitor = SystemMonitor()
        
        # 记录开始时间
        start_time = time.time()
        
        # 启动监控
        monitor.start()

        model = YOLO(r'/szzn/luoluo/work2/YOLOv8.2/消融实验/BDD_100K_300/7YOLOv8_SGBackbone_SlimNeck_DetectSA/train/exp/weights/best.pt') # select your model.pt path
        model.predict(source=r'/szzn/luoluo/work2/YOLOv8.2/datasets/BDD_100K/images/test',
                    # split='test',  # 使用测试集
                    imgsz=640,
                    name='exp',
                    save=True,
                    save_conf = True,
                    batch=64,
                    workers=8,
                    device='7',
                    # classes=0,
                    project= '/szzn/luoluo/work2/YOLOv8.2/消融实验/BDD_100K_300/7YOLOv8_SGBackbone_SlimNeck_DetectSA/test_jiankong',
                    )
        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 停止监控
        monitor.stop()
        time.sleep(1)  # 等待监控线程结束

        # 打印报告
        print_report(monitor, total_time)

    except ImportError as e:
        print(f"缺少依赖库: {e}\n请执行: pip install psutil")
    except Exception as e:
        print(f"运行错误: {str(e)}")
