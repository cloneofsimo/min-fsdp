# DeepSpeed introduces zero-memory fragments, which is to say they keep the parameters and other stuff in nicely flattened buffer.
# This codebase will introduce how that can be done with narrow, as well as keeping the buffer in the optimizer.

import torch
import math


class AdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = []
        self.v = []
        self.t = 0

        self.master_params = torch.cat([p.data.view(-1) for p in self.params]).clone()

        # Initialize moments
        self.m = torch.zeros_like(self.master_params)
        self.v = torch.zeros_like(self.master_params)

        # Update param data to point to the master param
        self.offsets = []
        offset = 0
        for param in self.params:
            numel = param.numel()
            self.offsets.append((offset, offset + numel))
            param.data = self.master_params.narrow(0, offset, numel).view_as(param)
            offset += numel

    def step(self):
        self.t += 1

        grad_flat = torch.cat(
            [
                (
                    p.grad.data.view(-1)
                    if p.grad is not None
                    else torch.zeros_like(p.data).view(-1)
                )
                for p in self.params
            ]
        )

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_flat
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_flat * grad_flat)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.master_params.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        # Copy updated master parameters to each parameter
        for i, param in enumerate(self.params):
            offset_start, offset_end = self.offsets[i]
            param.data = self.master_params.data[offset_start:offset_end].view_as(param)


from basic import DummyModel, check_model_from_reference

model = DummyModel()
optimizer = AdamOptimizer(model.parameters(), lr=4.0)

input = torch.randn(10, generator=torch.Generator().manual_seed(42))
target = torch.randn(1, generator=torch.Generator().manual_seed(42))

output = model(input)
loss = torch.nn.functional.mse_loss(output, target)

loss.backward()

optimizer.step()

check_model_from_reference(model)
