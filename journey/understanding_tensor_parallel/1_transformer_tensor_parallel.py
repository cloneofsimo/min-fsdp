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

from torch_profiler_utils import ContextManagers, get_torch_profiler

from pdb import set_trace as Tra


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

'''
adapted from karpathy
https://github.com/karpathy/nanoGPT/blob/master/model.py
'''
class Attention(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(Attention, self).__init__()
        assert hidden % nhead == 0, "hidden size should be divisible by nhead"
        self.dhead = hidden // nhead
        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias)
        self.v_proj = nn.Linear(hidden, hidden, bias=bias)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        k = self.k_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        v = self.v_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(x)

class MLP(nn.Module):
    def __init__(self, hidden, bias=False):
        super(MLP, self).__init__()
        self.ffn1 = nn.Linear(hidden, 4*hidden, bias)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(4*hidden, hidden, bias)

    def forward(self, x):
        return self.ffn2(self.act(self.ffn1(x)))

class LayerNorm(nn.Module):
    def __init__(self, hidden, bias=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden)) if bias else None

    def forward(self, x):
        return F.layer_norm(x.float(), self.weight.shape, self.weight, self.bias, 1e-5).type_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(ResidualBlock, self).__init__()
        self.ln1 = LayerNorm(hidden, bias)
        self.attn = Attention(hidden, nhead, bias)
        self.ln2 = LayerNorm(hidden, bias)
        self.mlp = MLP(hidden, bias)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, hidden, nhead, nlayer, bias=False):
        super(Transformer, self).__init__()
        assert bias == False, "currently bias is not supported"
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.model = nn.ModuleDict(
            dict(
                wte = nn.Embedding(vocab_size, hidden), # long tensor -> 3d tensor -> channel dim 쪼개
                wpe = nn.Embedding(block_size, hidden),
                h = nn.ModuleList([ResidualBlock(hidden, nhead, bias) for _ in range(nlayer)]),
                ln = LayerNorm(hidden, bias=bias),
            )
        )
        self.lm_head = nn.Linear(hidden, vocab_size, bias=bias)
        self.model.wte.weight = self.lm_head.weight # for pure megatron implementation, we automatically tie embedding 

    def compute_loss(self, z, y, ignore_index=-100, reduction='mean'):
        return F.cross_entropy(z, y, ignore_index=ignore_index, reduction=reduction)

    def forward(self, x, y): 
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.model.wte(x) + self.model.wpe(pos)
        for block in self.model.h:
            x = block(x)
        x = self.model.ln(x)
        z = self.lm_head(x).float() # projection to logit space and upcast
        z = z[..., :-1, :].contiguous().view(B*(T-1), -1) # B*T, C
        y = y.view(-1) # B*T, 1
        return self.compute_loss(z, y), z

class g(torch.autograd.Function):
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    def backward(ctx, dx):
        return dx

class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, dx):
        dist.all_reduce(dx, op=dist.ReduceOp.SUM)
        return dx

def rowwise_forward(self, x):
    bias = self.bias if self.bias else None
    x = F.linear(x, self.weight, bias)
    return g.apply(x)

def colwise_forward(self, x):
    bias = self.bias if self.bias else None
    x = f.apply(x)
    return F.linear(x, self.weight, bias)

'''
Refrences for vocab parallel (but it's not exactly same)
https://github.com/NVIDIA/Megatron-LM/blob/2d487b1871ba64ef1625781ea05715af1bc0d8ee/megatron/core/tensor_parallel/cross_entropy.py#L121-L126
https://github.com/NVIDIA/Megatron-LM/blob/e8f8e63f13a074f7e35d72c8bfb3e1168cd84e8e/megatron/core/tensor_parallel/layers.py#L151
https://github.com/pytorch/pytorch/blob/5ed3b70d09a4ab2a5be4becfda9dd0d3e3227c39/torch/distributed/tensor/parallel/loss.py#L126
https://github.com/pytorch/pytorch/blob/41e653456e4a96b43ea96c9cd3cddc63ea74711d/torch/ao/nn/qat/modules/embedding_ops.py#L11
https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py
'''

def get_mask_and_masked_input(x, vocab_start_index, vocab_end_index):
    x_mask = (x < vocab_start_index) | (x >= vocab_end_index)
    x = x.clone() - vocab_start_index
    x[x_mask] = 0
    return x, x_mask

class LossParallel_:
    def get_logit_max(z):
        return torch.max(z.float(), dim=-1)[0]

    def get_exp(z, z_max):
        z -= z_max.unsqueeze(dim=-1)
        exp = torch.exp(z) # B*T, C
        sum_exp = torch.sum(exp, dim=-1, keepdim=True) # B*T, 1
        return z, exp, sum_exp

    def get_one_hot(y, z, vocab_start_index, vocab_end_index):
        y, y_mask = get_mask_and_masked_input(y, vocab_start_index, vocab_end_index)
        y = F.one_hot(y, num_classes=z.size(1))
        y.masked_fill_(y_mask.unsqueeze(-1), 0.0)
        return y, y_mask

    def get_nll_loss(z, y, exp, sum_exp, y_one_hot, y_mask, ignore_index, reduction):
        # compute loss using log sum exponential trick # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        log_sum_exp = torch.log(sum_exp) # normalizer
        log_sum_exp.masked_fill_(y_mask.unsqueeze(-1), 0.0)
        gt_z = torch.sum(z * y_one_hot, dim=1)

        # Compute the loss
        divisor = 1 if reduction == 'sum' else (y!=ignore_index).sum()
        loss = (log_sum_exp.squeeze(1) - gt_z) / divisor
        loss = torch.where(y == ignore_index, torch.tensor(0.0, device=z.device), loss) # token-level loss
        loss = loss.sum()
        return loss, divisor

class LossParallel(torch.autograd.Function):
    def forward(ctx, z, y, vocab_start_index, vocab_end_index, ignore_index=-100, reduction='mean'):

        # communicate max logit value for numerical stability
        z_max = LossParallel_.get_logit_max(z) # B*T, C
        dist.all_reduce(z_max, op=dist.ReduceOp.MAX) # max

        # get numerical stable exponentiated vectors
        z, exp_z, sum_exp_z = LossParallel_.get_exp(z, z_max)
        dist.all_reduce(sum_exp_z, op=dist.ReduceOp.SUM)

        # compute loss and reduce all
        y_one_hot, y_mask = LossParallel_.get_one_hot(y, z, vocab_start_index, vocab_end_index)
        loss, divisor = LossParallel_.get_nll_loss(z, y, exp_z, sum_exp_z, y_one_hot, y_mask, ignore_index, reduction)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) # mean and sum loss

        # store results for backward
        ctx.save_for_backward(exp_z.div_(sum_exp_z), y_one_hot, divisor)
        return loss

    def backward(ctx, grad_output):
        y_hat, y_one_hot, divisor = ctx.saved_tensors
        dz = y_hat - y_one_hot # logit gradient 
        dz /= divisor # dL/dLogit
        dz *= grad_output # 1.0 because it's end 
        return dz, None, None, None, None, None # No gradients needed for y, ignore_index, or reduction parameters

def embedding_parallel(self, x):
    x, x_mask = get_mask_and_masked_input(x, self.vocab_start_index, self.vocab_end_index)
    x = F.embedding(x, self.weight)
    x.masked_fill_(x_mask.unsqueeze(-1), 0.0)
    return g.apply(x) # because readout layer is col-wise, embedding layer is row-wise

def parallelize_module(
    args,
    model: nn.Module, 
    world_size: int, 
    rank: int, 
):
    assert world_size > 1, "need at least two devices for TP"
    colwise_list = ['q_proj', 'k_proj', 'v_proj', 'ffn1']
    rowwise_list = ['o_proj', 'ffn2']

    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            parallelize_module(args, module, world_size, rank)

        '''
        pytorch impl matmul with transposed weight matrix,
        so you should slice weight matrix counter-intuitively. 
        '''
        for _ in rowwise_list:
            if _ in name.lower():
                assert module.weight.size(1) % world_size == 0 
                chunk_size = module.weight.size(1)//world_size
                module.weight.data = module.weight.data[:, chunk_size*rank: chunk_size*(rank+1)].contiguous()
                module.forward = rowwise_forward.__get__(module)
        for _ in colwise_list:
            if _ in name.lower():
                assert module.weight.size(0) % world_size == 0
                chunk_size = module.weight.size(0)//world_size
                module.weight.data = module.weight.data[chunk_size*rank: chunk_size*(rank+1), :].contiguous()
                module.forward = colwise_forward.__get__(module)

        '''
        you should slice embedding weight matrix col-wise (vocab dimension),
        so you need to perform softmax operation across sliced vocab dim.
        and because original megatron paper tie embedding and unembedding matrices, you should care this too.
        '''
        if args.loss_parallel:
            if 'lm_head' in name.lower() or 'wte' in name.lower():
                ## TODO: need vocab padding
                chunk_size = module.weight.size(0)//world_size
                vocab_start_index = chunk_size*rank
                vocab_end_index = chunk_size*(rank+1)

                if 'lm_head' in name.lower():
                    module.weight.data = module.weight.data[vocab_start_index:vocab_end_index, :].contiguous()
                    module.forward = colwise_forward.__get__(module)
                    def loss_parallel(x, y, ignore_index=-100, reduction='mean'):
                        return LossParallel.apply(x, y, vocab_start_index, vocab_end_index, ignore_index, reduction)
                    model.compute_loss = loss_parallel

                elif 'wte' in name.lower():
                    module.vocab_start_index = vocab_start_index
                    module.vocab_end_index = vocab_end_index
                    module.forward = embedding_parallel.__get__(module)

def get_dummy_input(
    vocab_size,
    device,
    batch_size=256,
    seq_len=1024,
):
    num_pad_tokens = seq_len//10
    input_ids = torch.randint(vocab_size, (batch_size, seq_len))
    labels = torch.cat((input_ids[:, 1:seq_len-num_pad_tokens], torch.full((batch_size, num_pad_tokens), -100)),1)
    return {
        'input_ids': input_ids.to(device),
        'labels': labels.to(device),
    }

def main(args):
    rank, world_size = init_dist()
    device = f"cuda:{rank}"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = args.hidden, 8, 2

    set_seed()
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to(device).train()
    if args.TP:
        assert model.nhead % world_size == 0, "nhead should be divisible by TP degree"
        parallelize_module(args, model, world_size, rank)
    else:
        if args.loss_parallel:
            def loss_parallel(x, y, ignore_index=-100, reduction='mean'):
                return LossParallel.apply(x, y, 0, vocab_size, ignore_index, reduction)
            model.compute_loss = loss_parallel
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    if args.batch_size and args.seq_len:
        input_ids = get_dummy_input(vocab_size-1, device, args.batch_size, args.seq_len)
    else:
        sent = "i love tensor parallelism."
        input_ids = tokenizer(sent, return_tensors='pt').to(device)
        input_ids['labels'] = input_ids['input_ids'][:, 1:]

    if args.use_torch_profiler:
        num_wait_steps, num_warmup_steps, num_active_steps, num_repeat = 1, 2, 3, 1
        num_iter = int((num_wait_steps + num_warmup_steps + num_active_steps)*num_repeat)
        context = [
            get_torch_profiler(
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                num_active_steps=num_active_steps,
                num_repeat=num_repeat,
                save_dir_name=f'TP_{args.TP}_world_size_{world_size}_hidden_{hidden}'
            )
        ]
    else:
        num_iter = 5
        context = []

    with ContextManagers(context) as p:
        for iter in range(num_iter):
            loss, z = model(input_ids['input_ids'], input_ids['labels'])
            z.retain_grad()
            loss.backward()

            message = f'''
            iter: {iter+1}
            input size: {input_ids['input_ids'].size()}
            num padding toekns: {(input_ids['labels'] == -100).sum()}
            loss: {loss}
            '''
            # message += f'''
            # z.grad: {z.grad}
            # '''

            optimizer.step()
            optimizer.zero_grad()

            print_message_with_master_process(rank, message)
            if args.use_torch_profiler:
                p.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--seq_len', default=None, type=int)

    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--TP', action='store_true')
    parser.add_argument('--loss_parallel', action='store_true')
    parser.add_argument('--use_torch_profiler', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)
