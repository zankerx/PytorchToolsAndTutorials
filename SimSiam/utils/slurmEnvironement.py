import os
import hostlist #pip install python-hostlist
import socket

'''
def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

'''

# get SLURM variables

def getJobId():
    
    return int(os.environ['SLURM_JOB_ID'])

def getRank():
    return int(os.environ['SLURM_PROCID'])

def getLocalRank():
    return int(os.environ['SLURM_LOCALID'])

def getWorldSize():
    return int(os.environ['SLURM_NTASKS'])

def getNumWorker():
    return int(os.environ['SLURM_CPUS_PER_TASK'])

def getHostnames():
    return hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

def getGpuId():
    return os.environ['SLURM_STEP_GPUS'].split(",")

def setupEnvironement():
    
    hostnames = getHostnames()
    gpu_ids = getGpuId()
    
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids)))
    os.environ['RANK'] = str(getRank())
    os.environ['WORLD_SIZE'] = str(getWorldSize())



