# So why is 3 not Zero-1 yet?
# 1. We need to do mixed-precision!
# 2. gradients could be in single bucket, we can use hooks to form gradients in unfragmented way.
# 3. minor details, such as skipping small params / checking for required_grad, and accepting param_group as input is all missing.


import torch
import torch.nn as nn
import torch.distributed as dist
import os
from basic import DummyModel, check_model_from_reference
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
        self.local_grad_buffer_hp = torch.zeros(current_offset).to(
            self.device, dtype=forward_dtype
        )

        for idx, (_, param) in enumerate(self._local_params()):
            si_s, si_e = self.shard_indices[idx]
            self.sharded_fp32_master_param[si_s:si_e] = param.data.view(-1).float()

            # set grad as well.
            param.grad = torch.zeros_like(param.data)
            param.grad.data = self.local_grad_buffer_hp[si_s:si_e].view_as(param.data)

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

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()

    @torch.compile()
    def adam_step(self):
        # Zero-1 Adam with compile!
        grad_fp = self.local_grad_buffer_hp.float()
        self.v.mul_(self.beta2).addcmul_(grad_fp, grad_fp, value=1 - self.beta2)
        self.m.mul_(self.beta1).add_(grad_fp, alpha=1 - self.beta1)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.sharded_fp32_master_param -= (
            self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        )

    def step(self):

        self.reduce_all_grads()

        dist.barrier()

        # These operation happens-per-device!
        self.t += 1.0

        self.adam_step()

        dist.barrier()

        # now sync the sharded_fp32_master_param to the actual model parameters.
        localidx = 0
        for idx, param in enumerate(self.params):
            to_send = torch.zeros_like(param.data.view(-1))

            if idx in self.local_param_indices:
                si_s, si_e = self.shard_indices[localidx]
                localidx += 1
                to_send = self.sharded_fp32_master_param[si_s:si_e].to(param.data.dtype)

            dist.broadcast(to_send, src=idx % self.world_size)
            param.data = to_send.view(param.data.size())


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def train_test():
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


class ConvNet(nn.Module):
    # for mnist
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def train_mnist():

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Running DDP example on rank {rank}, world size: {world_size}")
    dist.init_process_group(backend="nccl", init_method="env://")

    device = f"cuda:{rank}"

    model = ConvNet().to(device)
    forward_dtype = torch.bfloat16
    optimizer = Zero1AdamOptimizer(
        model.parameters(), lr=2e-3, forward_dtype=forward_dtype
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        root="./data", download=True, train=True, transform=transform
    )

    valset = datasets.MNIST(
        root="./data", download=True, train=False, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    for epoch in range(1):
        for i, (data, target) in enumerate(trainloader):
            data, target = data.to(device, dtype=forward_dtype), target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print0(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item()}")

    # eval
    model.eval()
    correct = 0
    total = 0
    valloader = DataLoader(valset, batch_size=64)
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device, dtype=forward_dtype), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print0(f"Accuracy: {correct/total}")


if __name__ == "__main__":
    train_mnist()
