# common things.

import torch
import math

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@torch.no_grad()
def check_model_from_reference(model):
    ref_state_dict = torch.load("model_ref.pth", map_location="cpu")
    model_state_dict = model.state_dict()

    for k in ref_state_dict.keys():
        ref = ref_state_dict[k].float()
        current = model_state_dict[k].cpu().float()
        assert torch.allclose(
            ref, current, atol=1e-2
        ), f"Model state dict does not match the reference model state dict for key {k}. Difference: {(ref - current).abs().max()}"

    print("Model state dict matches the reference model state dict")
