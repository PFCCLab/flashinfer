# Copyright (c) 2024 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch.distributed as dist


def pytest_sessionfinish(session, exitstatus):
    """Cleanup torch.distributed at the end of pytest session.

    This runs after all tests complete but before Python shutdown,
    avoiding the "destroy_process_group() was not called" warning.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


"""
Shared test utilities for comm tests.
"""

import ctypes
import os

from flashinfer.comm.mnnvl import MnnvlMemory


def _check_pidfd_permissions() -> bool:
    """Check if pidfd_getfd syscall is available and permitted.

    This is required for MNNVL in containers - the SYS_PTRACE capability
    must be available for cross-process file descriptor sharing.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
        SYS_pidfd_open = 434
        SYS_pidfd_getfd = 438

        # Try to open our own process and get our own fd
        my_pid = os.getpid()
        pidfd = syscall(SYS_pidfd_open, my_pid, 0)
        if pidfd < 0:
            return False

        # Try pidfd_getfd on stdin (fd=0) - this tests the permission
        # We don't actually need the result, just checking if it's permitted
        test_fd = syscall(SYS_pidfd_getfd, pidfd, 0, 0)
        os.close(pidfd)

        if test_fd < 0:
            err = ctypes.get_errno()
            if err == 1:  # EPERM - permission denied (container issue)
                return False
            # Other errors (like EBADF) are OK - permission check passed
        else:
            os.close(test_fd)

        return True
    except Exception:
        return False


def mnnvl_available() -> bool:
    """Check if MNNVL is fully available (hardware + container permissions)."""
    return MnnvlMemory.supports_mnnvl() and _check_pidfd_permissions()


def pytest_addoption(parser):
    parser.addoption("--num_nodes", type=int, default=1)
    parser.addoption("--node_id", type=int, default=0)
    parser.addoption("--dist_init_method", type=str, default="tcp://localhost:29501")


@pytest.fixture
def num_nodes(request):
    return request.config.getoption("--num_nodes")


@pytest.fixture
def node_id(request):
    return request.config.getoption("--node_id")


@pytest.fixture
def dist_init_method(request):
    return request.config.getoption("--dist_init_method")


# ---------------------------------------------------------------------------
# Paddle compat monkey-patches (para44-para48, para52)
# ---------------------------------------------------------------------------
import functools

import torch

# para44/para45/para52: assert_close bfloat16/float16 fix + Paddle wraps all errors
_orig_assert_close = torch.testing.assert_close


def _is_paddle_isclose_dtype_error(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur)
        # para52: Paddle wraps any assert_close internal error with this message
        if "resulted in the unexpected exception above" in msg:
            return True
        if ("bfloat16" in msg or "float16" in msg) and (
            "isclose" in msg or "NotFound" in msg
        ):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return False


def _manual_allclose(actual, expected, rtol, atol):
    a = actual.float().detach().cpu().numpy()
    e = expected.float().detach().cpu().numpy()
    diff = abs(a - e)
    tol = atol + rtol * abs(e)
    if not (diff <= tol).all():
        max_diff = float(diff.max())
        raise AssertionError(
            f"Tensors are not close! Max diff: {max_diff:.6f}, rtol={rtol}, atol={atol}"
        )


@functools.wraps(_orig_assert_close)
def _paddle_compat_assert_close(actual, expected, *args, **kwargs):
    try:
        _orig_assert_close(actual, expected, *args, **kwargs)
    except RuntimeError as e:
        if _is_paddle_isclose_dtype_error(e):
            rtol = kwargs.get("rtol")
            atol = kwargs.get("atol")
            dt = actual.dtype if isinstance(actual, torch.Tensor) else torch.float32
            if rtol is None:
                rtol = (
                    0.016
                    if dt == torch.bfloat16
                    else (0.001 if dt == torch.float16 else 1.3e-6)
                )
            if atol is None:
                atol = 1e-5
            _manual_allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            raise


torch.testing.assert_close = _paddle_compat_assert_close

# para46: torch.equal returns Tensor not bool in Paddle compat
_orig_equal = torch.equal


@functools.wraps(_orig_equal)
def _paddle_compat_equal(input, other):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        if input.shape != other.shape:
            return False
    result = _orig_equal(input, other)
    if isinstance(result, torch.Tensor):
        return bool(result.all().item()) if result.numel() > 1 else bool(result.item())
    return bool(result)


torch.equal = _paddle_compat_equal

# para47: tensor.multiply(scalar) -- Paddle compat may not accept Python scalar
_orig_tensor_multiply = torch.Tensor.multiply


def _paddle_compat_tensor_multiply(self, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    return _orig_tensor_multiply(self, other)


torch.Tensor.multiply = _paddle_compat_tensor_multiply

# para48: clamp_min / clamp_max missing on Tensor in Paddle compat
torch.Tensor.clamp_min = lambda self, v: torch.clamp(self, min=v)
torch.Tensor.clamp_max = lambda self, v: torch.clamp(self, max=v)
