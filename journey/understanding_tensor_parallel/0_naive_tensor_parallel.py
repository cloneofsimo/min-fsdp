import os
import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank: {rank}, world size: {world_size}")
    return rank, world_size

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

class DummyModel(torch.nn.Module):
    def __init__(self, hidden, bias=False):
        super(DummyModel, self).__init__()
        assert bias == False, "currently bias is not supported"
        self.fc1 = torch.nn.Linear(hidden, hidden, bias=bias) # for Colwise, 128, 128
        self.fc2 = torch.nn.Linear(hidden, hidden, bias=bias) # for Rowwise, 128, 128

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def colwise_backward(self, grad_output):
    grad_input = grad_output.mm(self.weight.t())
    dist.all_reduce(grad_input, op=dist.ReduceOp.SUM) # addmm
    return grad_input

def rowwise_forward(self, x):
    bias = self.bias if self.bias else None
    x = F.linear(x, self.weight, bias)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def parallelize_module(
    model: torch.nn.Module, 
    world_size: int, 
    rank: int, 
    layer_tp_plan: Dict
):
    assert world_size > 1, "need at least two devices for TP"

    for name, module in model.named_children():
        if name in layer_tp_plan:
            assert layer_tp_plan[name] in ['colwise', 'rowwise'], "plan should be colwise or rowwise"

            '''
            for example, weight of column wise parallel linear layer should be splitted into row-wise
            because pytorch implementation of linear layer is X = XW^T (F.linear(x, self.weight, bias))
            '''
            if layer_tp_plan[name] == 'rowwise':
                assert module.weight.size(1) % world_size == 0 
                chunk_size = module.weight.size(1)//world_size # e.g. world_size = 2, rank = 0, 1
                module.weight.data = module.weight.data[:, chunk_size*rank: chunk_size*(rank+1)].contiguous() # weight 128, 16 // input 10, 128
                module.forward = rowwise_forward.__get__(module)

            elif layer_tp_plan[name] == 'colwise':
                assert module.weight.size(0) % world_size == 0
                chunk_size = module.weight.size(0)//world_size
                module.weight.data = module.weight.data[chunk_size*rank: chunk_size*(rank+1), :].contiguous() # weight 16, 128  // input 10, 16
                module.backward = colwise_backward.__get__(module)


def main(args):
    rank, world_size = init_dist()
    device = f"cuda:{rank}"
    bsz, hidden = 8, 128
    num_iter, lr = 2, 0.01

    ## create model and parallelize if TP
    set_seed()
    model = DummyModel(hidden).to(device).train()
    if args.TP:
        layer_tp_plan = {
            "fc1": 'colwise',
            "fc2": 'rowwise',
        }
        parallelize_module(model, world_size, rank, layer_tp_plan)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    print_message_with_master_process(rank, f'model: {model}')

    ## create dummy input
    set_seed()
    x = torch.randn(bsz, hidden).to(device)

    ## for loop
    for iter in range(num_iter):
        output = model(x)
        loss = output.sum()
        loss.backward()

        ## get gathered gradient results
        if args.TP:
            fc1_grad = [torch.zeros_like(model.fc1.weight, dtype=torch.float32) for _ in range(world_size)]
            dist.all_gather(fc1_grad, model.fc1.weight.grad)
            fc1_grad = torch.cat(fc1_grad, dim=0)
        else:
            fc1_grad = model.fc1.weight.grad
        
        optimizer.step()
        optimizer.zero_grad()

        ## print outputs
        message = f'''
        iter: {iter+1}
        output: {output}
        loss: {loss}
        fc1_grad = {fc1_grad}
        '''
        print_message_with_master_process(rank, message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TP', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)