## Great! Now let's partition the master weight in full-precision, as well as the optimizer state.
# This would require torch.dist for communication,
# After backward, we would need to reduce the gradients to each shards
# Which needs to be updated per-shard.
# Note that this does not partition gradients, parameters, activation, instead, *just* the optimizer state.
# Thus this is Zero-1 algorithm (well not exactly but we are close)


import torch
import torch.distributed as dist
import os
from basic import DummyModel, check_model_from_reference


class NotYetButCloseZero1AdamOptimizer:
    def __init__(
        self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, skip_small_parameters=5
    ):
        self.params = list(
            params
        )  # fc1.weight.data, fc1.bias.data, fc2.weight.data, fc2.bias.data, final.weight.data
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.skip_small_parameters = skip_small_parameters  # this is not used.

        self.t = 0
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_world_size = self.world_size
        self.device = f"cuda:{self.local_rank}"
        self.offsets = []
        self.shard_indices = []

        self.param_flattened = torch.cat([param.data.view(-1) for param in self.params])

        # 2M
        offset = 0
        for param in self.params:
            # view flatten
            param.data = self.param_flattened[
                offset : offset + param.data.numel()
            ].view_as(param.data)
            offset += param.data.numel()

        # 1M, 1M
        current_offset = 0
        # Initialize config per-shard.
        for _, param in self._local_params():
            self.offsets.append(param.data.view(-1).size(0))
            self.shard_indices.append(
                (current_offset, current_offset + param.data.view(-1).size(0))
            )
            current_offset += param.data.view(-1).size(0)

        self.v = torch.zeros(current_offset).to(self.device)
        self.m = torch.zeros(current_offset).to(self.device)
        self.sharded_fp32_master_param = torch.zeros(current_offset).to(self.device)
        self.local_grad_buffer = torch.zeros(current_offset).to(self.device)

        for idx, (_, param) in enumerate(self._local_params()):
            si_s, si_e = self.shard_indices[idx]
            self.sharded_fp32_master_param[si_s:si_e] = param.data.view(-1)

        dist.barrier()
        print(
            f"Rank {self.local_rank}",
            self.sharded_fp32_master_param.shape,
            self.shard_indices,
        )
        dist.barrier()

    def _local_params(self):

        # iterator that returns set of params this rank is responsible of.
        idx = 0
        for param in self.params:
            if idx % self.local_world_size == self.local_rank:
                yield idx, param  # 0, fc1.weight.data, 2, fc2.weight.data,
            idx += 1

    def reduce_all_grads(self):
        for param in self.params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def step(self):

        self.reduce_all_grads()

        dist.barrier()

        for idx, (_, param) in enumerate(self._local_params()):
            si_s, si_e = self.shard_indices[idx]
            self.local_grad_buffer[si_s:si_e] = param.grad.data.view(-1)

        print(f"Rank: {self.local_rank}", self.local_grad_buffer.shape)
        # These operation happens-per-device!

        self.t += 1
        self.v.mul_(self.beta2).addcmul_(
            self.local_grad_buffer, self.local_grad_buffer, value=1 - self.beta2
        )
        self.m.mul_(self.beta1).add_(self.local_grad_buffer, alpha=1 - self.beta1)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.sharded_fp32_master_param -= (
            self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        )

        dist.barrier()

        # now sync the sharded_fp32_master_param to the actual model parameters.
        localidx = 0
        for idx, param in enumerate(self.params):
            to_send = param.data.view(-1)

            if idx % self.local_world_size == self.local_rank:
                si_s, si_e = self.shard_indices[localidx]
                localidx += 1
                to_send = self.sharded_fp32_master_param[si_s:si_e]

            dist.broadcast(to_send, src=idx % self.world_size)
            param.data = to_send.view(param.data.size())


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def train():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Running DDP example on rank {rank}, world size: {world_size}")

    dist.init_process_group(backend="nccl", init_method="env://")

    device = f"cuda:{rank}"

    model = DummyModel().to(device)

    optimizer = NotYetButCloseZero1AdamOptimizer(model.parameters(), lr=1e-1)

    input = torch.randn(10, generator=torch.Generator().manual_seed(42)).to(device)
    target = torch.randn(1, generator=torch.Generator().manual_seed(42)).to(device)

    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)

    loss.backward()
    optimizer.step()

    check_model_from_reference(model)  # Nice!


if __name__ == "__main__":
    train()
