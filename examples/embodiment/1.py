import torch
from torch.multiprocessing import Process, Queue

def worker(q):
    t = torch.zeros(1024, device="cuda")
    q.put(t.storage())  # 或者其他共享 CUDA storage 的方式

if __name__ == "__main__":
    q = Queue()
    p = Process(target=worker, args=(q,))
    p.start()
    storage = q.get()  # 这里内部会触发 rebuild_cuda_tensor/_new_shared_cuda
    p.join()
