import torch
import gc

def free_cuda_memory():
    torch.cuda.empty_cache()       # Releases all unused memory from PyTorch's caching allocator
    gc.collect()                   # Runs Python garbage collector
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.ipc_collect()  # Cleans up interprocess communication handles
    print("✅ CUDA memory freed.")

# Usage
free_cuda_memory()
