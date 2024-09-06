
import os
import torch
import socket
from datetime import datetime, timedelta
from contextlib import contextmanager, ExitStack
from typing import Any, ContextManager, Iterable, List, Tuple


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        entered_contexts = [
            self.stack.enter_context(cm) for cm in self.context_managers
        ]
        # Assuming you want to return the first context manager, adjust as needed
        return entered_contexts[0] if entered_contexts else None

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def get_torch_profiler(
    use_tensorboard=True,
    root_dir="./assets/torch_profiler_log",
    save_dir_name="tmp",

    num_wait_steps=1, # During this phase profiler is not active.
    num_warmup_steps=2, # During this phase profiler starts tracing, but the results are discarded.
    num_active_steps=2, # During this phase profiler traces and records data.
    num_repeat=1,  # Specifies an upper bound on the number of cycles.

    record_shapes=True,
    profile_memory=True,
    
    with_flops=True,
    with_stack = False, # Enable stack tracing, adds extra profiling overhead. stack tracing adds an extra profiling overhead.
    with_modules=True,
):
    save_path=os.path.join(root_dir, save_dir_name)
    os.makedirs(save_path, exist_ok=True)

    '''
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs
    https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    https://github.com/pytorch/kineto/blob/main/tb_plugin/README.md
    https://oss.navercorp.com/seunghyun-seo1/seosh_fairseq/blob/main/toward_iclr/cuda_profile_speech_encoder.py
    
    https://pytorch.org/blog/accelerating-generative-ai-2/
    https://www.deepspeed.ai/tutorials/pytorch-profiler/
    https://ui.perfetto.dev
    chrome://tracing/

    https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/
    https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    https://pytorch.org/blog/pytorch-profiler-1.9-released/

    231214 added
    https://pytorch.org/blog/understanding-gpu-memory-1/
    https://github.com/pytorch/pytorch.github.io/tree/site/assets/images/understanding-gpu-memory-1
    '''
    
    def trace_handler(prof: torch.profiler.profile):
        TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        file_prefix = f"{host_name}_{timestamp}"
        prof.export_chrome_trace(f"{save_path}/{file_prefix}.json.gz")

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        
        with_flops=with_flops,
        with_stack = with_stack,
        with_modules = with_modules,

        schedule=torch.profiler.schedule(
            wait=num_wait_steps,
            warmup=num_warmup_steps,
            active=num_active_steps,
            repeat=num_repeat,
        ),
        on_trace_ready = trace_handler if not use_tensorboard else torch.profiler.tensorboard_trace_handler(save_path),
    )