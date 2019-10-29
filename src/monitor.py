# import resource
from threading import Thread
import GPUtil
import cpuinfo
import psutil
import time
import os
from livelossplot import PlotLosses


DEVICE_ID_LIST = GPUtil.getAvailable(
    order="memory", limit=1
)  # get the fist gpu with the lowest load


def get_gpu_stat():
    gpu_ids = GPUtil.getAvailable(limit=10)
    for gpu_id in gpu_ids:
        GPU = GPUtil.getGPUs()[gpu_id]
        GPU_load = GPU.load * 100
        GPU_memoryUtil = GPU.memoryUtil / 2.0 ** 10
        GPU_memoryTotal = GPU.memoryTotal / 2.0 ** 10
        GPU_memoryUsed = GPU.memoryUsed / 2.0 ** 10
        GPU_memoryFree = GPU.memoryFree / 2.0 ** 10
        print("Current GPU (ID:%d) name:%s" % (gpu_id, GPU.name))
        print(
            "-- total_GPU_memory: %.3fGB;init_GPU_memoryFree:%.3fGB init_GPU_load:%.3f%% GPU_memoryUtil:%d%% GPU_memoryUsed:%.3fGB"
            % (
                GPU_memoryTotal,
                GPU_memoryFree,
                GPU_load,
                GPU_memoryUtil,
                GPU_memoryUsed,
            )
        )


def get_cpu_stat():
    print(
        "Cpu count:%d brand:%s avg_load_1:%.3f%% avg_load_5:%.3f%% avg_load_15:%.3f%%"
        % (
            os.cpu_count(),
            cpuinfo.get_cpu_info()["brand"],
            os.getloadavg()[0],
            os.getloadavg()[1],
            os.getloadavg()[2],
        )
    )


def get_mem_stat():
    # Main memory info
    memoryInfo = (
        psutil.virtual_memory()
    )  # svmem(total, available, percent, used, free, active, inactive, buffers, cached, shared, slab)
    print(
        "Current RAM -- total:%.3fGB, available:%.3fGB, used:%.3f GB,free:%.3fGB"
        % (
            memoryInfo.total / 2.0 ** 30,
            memoryInfo.available / 2.0 ** 30,
            memoryInfo.used / 2.0 ** 30,
            memoryInfo.free / 2.0 ** 30,
        )
    )


# print current devices available
def devices_status():
    get_cpu_stat()
    get_mem_stat()
    get_gpu_stat()


class Monitor(Thread):
    def __init__(self, delay=1, GPU_id=0, pid=os.getpid(), verbose=False,live_draw=True):
        super(Monitor, self).__init__()
        self.live_update=False  # if true, a live plot will update
        if len(DEVICE_ID_LIST) < 1 or GPU_id == None:
            self.hasgpu = False
        else:
            self.hasgpu = True
        self.GPU_id = GPU_id
        self.start_time = time.time()  # Start time
        self.verbose = verbose  # if update the usage status during the process
        self.live_draw = live_draw
        self.stopped = False  # flag for stop the monitor
        self.delay = delay  # Time between calls to GPUtil
        print("Start monitoring resources...")
        # CPU info
        self.pid = pid  # Process_ID
        self.CPU_load = psutil.Process(self.pid).cpu_percent(
            interval=1
        )  # CPU load %,the max of load is core_number*100%
        self.CPU_load_max = self.CPU_load
        self.CPU_load_avg = self.CPU_load

        self.count = 1  # Count for calculate the average usage
        if self.hasgpu:
            print("Process_ID:%d GPU_ID:%d" % (pid, GPU_id))
        print(
            "CPU name:%s \nCPU Core number:%d"
            % (cpuinfo.get_cpu_info()["brand"], psutil.cpu_count())
        )

        # Main memory info
        self.memoryInfo = (
            psutil.virtual_memory()
        )  # svmem(total, available, percent, used, free, active, inactive, buffers, cached, shared, slab)

        self.memoryUsed = (
            psutil.Process(self.pid).memory_info()[0] / 2.0 ** 30
        )  # current app memory use in GB
        self.memoryUsed_max = self.memoryUsed
        self.memoryUsed_avg = self.memoryUsed

        print(
            "Current RAM (PID:%d) -- total:%.3fGB, available:%.3fGB, used:%.3f GB,free:%.3fGB"
            % (
                self.pid,
                self.memoryInfo.total / 2.0 ** 30,
                self.memoryInfo.available / 2.0 ** 30,
                self.memoryInfo.used / 2.0 ** 30,
                self.memoryInfo.free / 2.0 ** 30,
            )
        )
        # GPU info
        if self.hasgpu:
            self.GPU_id = GPU_id
            self.GPU = GPUtil.getGPUs()[GPU_id]
            self.GPU_load = self.GPU.load * 100
            self.GPU_load_max = self.GPU_load
            self.GPU_load_avg = self.GPU_load
            self.GPU_memoryUtil = self.GPU.memoryUtil / 2.0 ** 10
            self.GPU_memoryUtil_max = self.GPU_memoryUtil
            self.GPU_memoryUtil_avg = self.GPU_memoryUtil
            self.GPU_memoryTotal = self.GPU.memoryTotal / 2.0 ** 10
            self.GPU_memoryUsed = self.GPU.memoryUsed / 2.0 ** 10
            self.GPU_memoryUsed_max = self.GPU_memoryUsed
            self.GPU_memoryUsed_avg = self.GPU_memoryUsed
            self.GPU_memoryFree = self.GPU.memoryFree / 2.0 ** 10
            print("GPU name:", self.GPU.name)
            print(
                "Current GPU (ID:%d)-- total_GPU_memory: %.3fGB;init_GPU_memoryFree:%.3fGB init_GPU_load:%.3f%% GPU_memoryUtil:%d%% GPU_memoryUsed:%.3fGB"
                % (
                    GPU_id,
                    self.GPU_memoryTotal,
                    self.GPU_memoryFree,
                    self.GPU_load,
                    self.GPU_memoryUtil,
                    self.GPU_memoryUsed,
                )
            )
        self.start()

    def run(self):
        while not self.stopped:
            
            if self.live_update==True and self.live_draw:
                self.liveloss.draw()
                self.live_update=False
            
            self.count += 1
            current_CPU_load = psutil.Process(self.pid).cpu_percent(interval=1)
            self.CPU_load_avg += current_CPU_load
            current_memoryUsed = (
                psutil.Process(self.pid).memory_info()[0] / 2.0 ** 30
            )  # memory use in GB
            self.memoryUsed_avg += current_memoryUsed

            flag = False
            if current_CPU_load > self.CPU_load_max:
                flag = True
                self.CPU_load_max = current_CPU_load
            if current_memoryUsed > self.memoryUsed_max:
                flag = True
                self.memoryUsed_max = current_memoryUsed
            if flag == True and self.verbose == True:
                print(
                    "Memory or CPU peaked (PID:%d) -- memory use:%.3fGB, CPU load:%.3f%%"
                    % (self.pid, self.memoryUsed_max, self.CPU_load_max)
                )
                flag = False

            if self.hasgpu:
                current_GPU_load = GPUtil.getGPUs()[self.GPU_id].load * 100
                self.GPU_load_avg += current_GPU_load
                current_GPU_memoryUtil = (
                    GPUtil.getGPUs()[self.GPU_id].memoryUtil * 100
                )  # gpu memory use in GB
                self.GPU_memoryUtil_avg += current_GPU_memoryUtil
                current_GPU_memoryUsed = (
                    GPUtil.getGPUs()[self.GPU_id].memoryUsed / 2.0 ** 10
                )  # gpu memory use in GB
                self.GPU_memoryUsed_avg += current_GPU_memoryUsed
                if current_GPU_load > self.GPU_load_max:
                    flag = True
                    self.GPU_load_max = current_GPU_load
                if current_GPU_memoryUtil > self.GPU_memoryUtil_max:
                    flag = True
                    self.GPU_memoryUtil_max = current_GPU_memoryUtil
                if current_GPU_memoryUsed > self.GPU_memoryUsed_max:
                    flag = True
                    self.GPU_memoryUsed_max = current_GPU_memoryUsed
                time.sleep(self.delay)
            if flag == True and self.verbose == True and self.hasgpu:
                print(
                    "GPU peaked -- memoryTotal:%.3fGB init_memoryFree:%.3fGB max_load:%.3f%% max_memoryUtil:%.3fGB max_memoryUsed:%.3fGB "
                    % (
                        self.GPU_memoryTotal,
                        self.GPU_memoryFree,
                        self.GPU_load_max,
                        self.GPU_memoryUtil_max,
                        self.GPU_memoryUsed_max,
                    )
                )
                flag = False
    def update_live_plot(self,logs):
        self.liveloss.update(logs)
        self.live_update=True
    
    def init_live_plot(self,file):
        self.liveloss = PlotLosses(fig_path=file)

    def stop(self):
        running_time = time.time() - self.start_time
        print("Program running time:%d seconds" % (running_time))
        if self.hasgpu:
            print(
                "Final Resource Usage Peak    -- CPU_load_max:%.3f%% memoryUsed_max:%.3fGB GPU_load_max:%.3f%% GPU_memoryUsed_max:%.3fGB GPU_memoryUtil_max:%d%%"
                % (
                    self.CPU_load_max,
                    self.memoryUsed_max,
                    self.GPU_load_max,
                    self.GPU_memoryUsed_max,
                    self.GPU_memoryUtil_max,
                )
            )
            print(
                "Final Resource Usage Average -- CPU_load_avg:%.3f%% memoryUsed_avg:%.3fGB GPU_load_avg:%.3f%% GPU_memoryUsed_avg:%.3fGB GPU_memoryUtil_avg:%d%%"
                % (
                    float(self.CPU_load_avg) / self.count,
                    float(self.memoryUsed_avg) / self.count,
                    float(self.GPU_load_avg) / self.count,
                    float(self.GPU_memoryUsed_avg) / self.count,
                    float(self.GPU_memoryUtil_avg) / self.count,
                )
            )
        else:
            print(
                "Final Resource Usage Peak    -- CPU_load_max:%.3f%% memoryUsed_max:%.3fGB "
                % (self.CPU_load_max, self.memoryUsed_max)
            )
            print(
                "Final Resource Usage Average -- CPU_load_avg:%.3f%% memoryUsed_avg:%.3fGB "
                % (
                    float(self.CPU_load_avg) / self.count,
                    float(self.memoryUsed_avg) / self.count,
                )
            )
        # memoryUseMax = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2.**20
        # print("check memoryUseMax:%.3fGB"%(memoryUseMax))
        self.stopped = True
        return running_time