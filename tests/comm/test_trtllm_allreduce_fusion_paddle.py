import socket
import pytest
import paddle
import paddle.distributed as dist_pp

paddle.enable_compat()
from paddle.device.cuda.graphs import CUDAGraph
import flashinfer.comm as comm

import os
import numpy as np

# test parameters
token_num = 128
hidden_dim = 1024
dtype = paddle.float16
pattern_code = comm.AllReduceFusionPattern.kAllReduce
layout_code = comm.QuantizationSFLayout.LINEAR
launch_with_pdl = False
use_oneshot = True
trigger_completion_at_end = True
fp32_acc = False


def kernel(workspace_tensor, rank, world_size):
    device = f"cuda:{rank}"
    message_size = token_num * hidden_dim
    dtype = paddle.float16
    # Create input data
    allreduce_in = paddle.randn(message_size, dtype=dtype, device=device)
    # allreduce_in_clone = allreduce_in.clone()
    all_reduce_out = paddle.zeros(message_size, dtype=dtype, device=device)

    # Add missing required parameters
    residual_in = paddle.randn(message_size, dtype=dtype, device=device)
    residual_out = paddle.zeros(message_size, dtype=dtype, device=device)
    norm_out = paddle.zeros(message_size, dtype=dtype, device=device)
    quant_out = paddle.zeros(message_size, dtype=dtype, device=device)
    scale_out = paddle.zeros(message_size // 16, dtype=dtype, device=device)
    rms_gamma = paddle.randn(hidden_dim, dtype=dtype, device=device)
    rms_eps = 1e-3
    scale_factor = paddle.tensor(0.5, dtype=paddle.float32, device=device)

    # Run fusion operation
    comm.trtllm_allreduce_fusion(
        allreduce_in=allreduce_in,
        world_size=world_size,
        world_rank=rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_tensor,
        launch_with_pdl=launch_with_pdl,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=pattern_code,
        allreduce_out=all_reduce_out,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
    )

    # paddle.cuda.synchronize()

    return allreduce_in, all_reduce_out


def _run_simple_worker(world_size, rank, distributed_init_port):
    # Create workspace
    # paddle.compat.enable_torch_proxy()
    # Set all required environment variables
    os.environ["FLAGS_SELECTED_GPUS"] = str(rank)  # Key: set GPU ID
    os.environ["PADDLE_TRAINER_ID"] = str(rank)
    os.environ["PADDLE_TRAINERS_NUM"] = str(world_size)
    os.environ["PADDLE_RANK_IN_NODE"] = str(rank)

    # Build endpoint list
    endpoints = ",".join(
        [f"127.0.0.1:{distributed_init_port + i + 10}" for i in range(world_size)]
    )
    os.environ["PADDLE_TRAINER_ENDPOINTS"] = endpoints
    os.environ["PADDLE_CURRENT_ENDPOINT"] = (
        f"127.0.0.1:{distributed_init_port + rank + 10}"
    )

    # Set NCCL related environment variables (optional but recommended)
    os.environ["FLAGS_SYNC_NCCL_ALLREDUCE"] = "1"

    # Set device
    paddle.set_device(f"gpu:{rank}")

    # Initialize distributed environment
    dist_pp.init_parallel_env()
    group_pp = dist_pp.get_group()

    try:
        # Create workspace
        ipc_handles, workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                token_num,
                hidden_dim,
                group=group_pp,
                use_fp32_lamport=False,
            )
        )

        dist_pp.barrier(group=group_pp)

        # Run fusion operation
        loop = 5
        s = paddle.cuda.Stream()
        s.wait_stream(paddle.cuda.current_stream())
        with paddle.cuda.stream(s):
            for _ in range(loop):
                allreduce_in_clone, all_reduce_out = kernel(
                    workspace_tensor, rank, world_size
                )

        g = CUDAGraph()
        g.capture_begin()
        for _ in range(loop):
            allreduce_in_clone, all_reduce_out = kernel(
                workspace_tensor, rank, world_size
            )
        g.capture_end()

        g.replay()
        paddle.cuda.synchronize()

        # # Calculate reference result
        dist_pp.all_reduce(allreduce_in_clone, group=group_pp)
        ref_allreduce_out = allreduce_in_clone.clone()

        # # Verify results
        tolerance = 8e-2
        np.testing.assert_allclose(
            all_reduce_out.numpy(), ref_allreduce_out.numpy(), atol=tolerance, rtol=1e-2
        )

        print(f"Rank {rank}: Test passed!")

    finally:
        dist_pp.barrier(group=group_pp)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce(ipc_handles, group=group_pp)
        dist_pp.destroy_process_group(group=group_pp)


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_trtllm_allreduce_fusion_simple():
    # Fixed test parameters
    world_size = 2

    paddle.manual_seed(42)
    paddle.cuda.manual_seed_all(42)

    available_gpus = paddle.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Requires {world_size} GPUs, but only {available_gpus} available")

    distributed_init_port = get_open_port()
    rank = dist_pp.get_rank()
    _run_simple_worker(world_size, rank, distributed_init_port)

    print("Simple allreduce fusion test: passed")


# test cmd: python -m paddle.distributed.launch --log_dir=log --devices=0,1
#  ./test_torch_pp_launch.py
if __name__ == "__main__":
    test_trtllm_allreduce_fusion_simple()
