# this implements referecne for Adam in pure pytorch.

import torch
import math
from basic import DummyModel, check_model_from_reference


class AdamOptimizer:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = torch.zeros_like(param.grad)


model = DummyModel()
optimizer = AdamOptimizer(model.parameters(), lr=4.0)

input = torch.randn(10, generator=torch.Generator().manual_seed(42))
target = torch.randn(1, generator=torch.Generator().manual_seed(42))
output = model(input)

loss = torch.nn.functional.mse_loss(output, target)

loss.backward()

optimizer.step()

check_model_from_reference(model)


print(model.fc2.bias.grad.data)
optimizer.zero_grad()
print(model.fc2.bias.grad.data)

# make a train loop
for i in range(1000):
    optimizer.zero_grad()
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Loss at iteration {i}: {loss.item()}")
