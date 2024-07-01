import torch
import torch.distributed as dist
import os


def init_process(rank, size, fn, backend="gloo"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run(rank, size):

    tensor = torch.tensor([rank]).float()
    tensor_list = [torch.zeros(1).float() for _ in range(size)]
    dist.all_gather(tensor_list, tensor)
    print(f"Rank {rank} gathered data {tensor_list}")


if __name__ == "__main__":
    size = 4  # Number of processes
    processes = []

    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
