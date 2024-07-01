import torch
import torch.distributed as dist
import os


def init_process(rank, size, fn, backend="nccl"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run(rank, size):

    tensor = torch.ones(1) * rank
    tensor = tensor.to(f"cuda:{rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} has data {tensor[0]}")


if __name__ == "__main__":
    size = 4  # Number of processes
    processes = []

    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
