import os
import hostlist #pip install python-hostlist


'''
tools :
    Slurm environment variables



'''

# Slurm environement variables
class Slurm():

    @staticmethod
    def getJobId():
        return int(os.environ['SLURM_JOB_ID'])

    @staticmethod
    def getRank():
        return int(os.environ['SLURM_PROCID'])

    @staticmethod
    def getLocalRank():
        return int(os.environ['SLURM_LOCALID'])

    @staticmethod
    def getWorldSize():
        return int(os.environ['SLURM_NTASKS'])

    @staticmethod
    def getNumWorker():
        return int(os.environ['SLURM_CPUS_PER_TASK'])

    @staticmethod
    def getHostnames():
        return hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    @staticmethod
    def getGpuId():
        return os.environ['SLURM_STEP_GPUS'].split(",")

    @staticmethod
    def setMasterPortAddr():
            
        hostnames = Slurm.getHostnames()
        gpu_ids = Slurm.getGpuId()
        
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids)))


