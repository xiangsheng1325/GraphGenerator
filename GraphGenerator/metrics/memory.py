import torch, os

## return current gpu memory cached
# torch.cuda.memory_reserved()
## return peak gpu memory cached
# torch.cuda.max_memory_reserved()
## reset peak gpu memory cached
# torch.cuda.reset_peak_memory_stats()

def get_peak_gpu_memory(device='cuda:0'):
    """
    :return: maximum memory cached (Byte)
    """
    return torch.cuda.max_memory_reserved(device)


def flush_cached_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_memory_usage():
    flush_cached_gpu_memory()
    current_memory = get_peak_gpu_memory()//1024
    print("Current gpu memory cached: {} KiB".format(current_memory))
    flush_cached_gpu_memory()
    a = torch.ones(3,3).cuda()
    print("Add a tensor to gpu.")
    current_memory = get_peak_gpu_memory() // 1024
    print("Current gpu memory cached: {} KiB".format(current_memory))
    del a
    print("Delete a tensor from gpu.")
    flush_cached_gpu_memory()
    current_memory = get_peak_gpu_memory() // 1024
    print("Current gpu memory cached: {} KiB".format(current_memory))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_memory_usage()