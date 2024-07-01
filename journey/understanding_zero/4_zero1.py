# So why is 3 not Zero-1 yet?
# 1. We need to do mixed-precision!
# 2. gradients could be in better form, we can use hooks to form gradients in unfragmented way.
# 3. minor details, such as skipping small params / checking for required_grad, and accepting param_group as input is all missing.


import torch
import torch.distributed as dist
import os
from basic import DummyModel, check_model_from_reference


class Zero1AdamOptimizer:
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        skip_small_parameters=5,
        forward_dtype=torch.bfloat16,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.skip_small_parameters = skip_small_parameters

        self.t = 0
        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_world_size = self.world_size
        self.device = f"cuda:{self.local_rank}"
        self.offsets = []
        self.shard_indices = []
        self.local_param_indices = set()

        self.param_flattened = torch.cat([param.data.view(-1) for param in self.params])
        offset = 0
        for param in self.params:
            # view flatten
            param.data = (
                self.param_flattened[offset : offset + param.data.numel()]
                .view_as(param.data)
                .to(device=self.device, dtype=forward_dtype)
            )
            offset += param.data.numel()

        current_offset = 0
        # Initialize config per-shard.

        for gidx, param in self._local_params():
            self.offsets.append(param.data.view(-1).size(0))
            self.shard_indices.append(
                (current_offset, current_offset + param.data.view(-1).size(0))
            )
            current_offset += param.data.view(-1).size(0)
            self.local_param_indices.add(gidx)

        self.v = torch.zeros(current_offset).to(self.device)
        self.m = torch.zeros(current_offset).to(self.device)
        self.sharded_fp32_master_param = torch.zeros(current_offset).to(self.device)
        self.local_grad_buffer = torch.zeros(current_offset).to(self.device)

        for idx, (_, param) in enumerate(self._local_params()):
            si_s, si_e = self.shard_indices[idx]
            self.sharded_fp32_master_param[si_s:si_e] = param.data.view(-1).float()

    def _local_params(self):
        # iterator that returns set of params this rank is responsible of.
        idx = 0
        for param in self.params:
            if not param.requires_grad:
                continue
            if (
                idx % self.local_world_size == self.local_rank
                or param.numel() < self.skip_small_parameters
            ):
                yield idx, param
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
            self.local_grad_buffer[si_s:si_e] = param.grad.data.view(-1).float()

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

            if idx in self.local_param_indices:
                si_s, si_e = self.shard_indices[localidx]
                localidx += 1
                to_send = self.sharded_fp32_master_param[si_s:si_e].to(param.data.dtype)

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
    forward_dtype = torch.bfloat16
    optimizer = Zero1AdamOptimizer(
        model.parameters(), lr=1e-1, forward_dtype=forward_dtype
    )

    input = torch.randn(10, generator=torch.Generator().manual_seed(42)).to(
        device, dtype=forward_dtype
    )
    target = torch.randn(1, generator=torch.Generator().manual_seed(42)).to(
        device, dtype=forward_dtype
    )

    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)

    loss.backward()
    optimizer.step()

    check_model_from_reference(model)  # Nice!


if __name__ == "__main__":
    train()
