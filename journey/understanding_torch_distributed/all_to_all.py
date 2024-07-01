import torch
import torch.distributed as dist
import os


def init_process(rank, size, fn, backend="nccl"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run(rank, size):

    send_tensors = [
        torch.ones(1).to(f"cuda:{rank}") * (rank * size + i) for i in range(size)
    ]
    recv_tensors = [torch.zeros(1).to(f"cuda:{rank}") for _ in range(size)]
    dist.all_to_all(recv_tensors, send_tensors)
    print(f"Rank {rank} sent {send_tensors} and received {recv_tensors}")


if __name__ == "__main__":
    size = 4  # Number of processes
    processes = []

    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
