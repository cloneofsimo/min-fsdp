from basic import DummyModel
from torch.optim import Adam
import torch

model = DummyModel()
optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-8)
print("Initial model parameters:", list(model.parameters()))
input = torch.randn(10, generator=torch.Generator().manual_seed(42))
target = torch.randn(1, generator=torch.Generator().manual_seed(42))
output = model(input)

loss = torch.nn.functional.mse_loss(output, target)

loss.backward()

optimizer.step()

print("Updated model parameters:", list(model.parameters()))

# write down
torch.save(model.state_dict(), "model_ref.pth")
