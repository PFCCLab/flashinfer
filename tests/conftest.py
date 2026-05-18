import json
import os
import types
from pathlib import Path
from typing import Any, Dict, Set

import paddle

paddle.enable_compat()
import pytest
import torch
# from torch.torch_version import TorchVersion
# §38: Paddle Event has no wait() method; PyTorch event.wait(stream=None)
# makes current/given stream wait for the event → use stream.wait_event(event)
def _paddle_event_wait(self, stream=None):
    if stream is None:
        stream = torch.cuda.current_stream()
    stream.wait_event(self)
paddle.device.Event.wait = _paddle_event_wait
# ss39: Paddle view() fails for non-contiguous -> fallback to reshape
_orig_view = torch.Tensor.view
def _paddle_compat_view(self, *args):
    if len(args) == 1 and isinstance(args[0], torch.dtype):
        try:
            return _orig_view(self, args[0])
        except Exception:
            return _orig_view(self.contiguous(), args[0])
    try:
        return _orig_view(self, *args)
    except (ValueError, RuntimeError):
        return self.reshape(*args)
torch.Tensor.view = _paddle_compat_view
# ss40: Paddle missing float8 set_value_with_tensor -> workaround via uint8
_orig_setitem = torch.Tensor.__setitem__
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
def _paddle_compat_setitem(self, idx, value):
    if self.dtype in _FP8_DTYPES and isinstance(value, torch.Tensor):
        try:
            self_u8 = _orig_view(self.contiguous(), torch.uint8)
            val_u8 = value.contiguous().view(torch.uint8)
            _orig_setitem(self_u8, idx, val_u8)
            return
        except Exception:
            pass
    _orig_setitem(self, idx, value)
torch.Tensor.__setitem__ = _paddle_compat_setitem

# ss41: Skip tests that need cubin download from NVIDIA servers
# (edge.urm.nvidia.com) which is unreachable in air-gapped environments.
# Strategy: patch cubin_loader.download_file to fail immediately (no hang)
# and also patch trtllm_fp8_block_scale_moe at Python level.
# For tests that fail because the C-level cubin callback returns no cubin,
# we set FLASHINFER_NO_DOWNLOAD to make get_artifact raise immediately.
import socket as _socket
import functools as _functools
import os as _os

def _check_nvidia_cubin_server(host="edge.urm.nvidia.com", port=443, timeout=3):
    """Return True if the NVIDIA cubin download server is reachable."""
    try:
        _socket.setdefaulttimeout(timeout)
        with _socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, _socket.timeout):
        return False

_NVIDIA_CUBIN_SERVER_REACHABLE = _check_nvidia_cubin_server()

if not _NVIDIA_CUBIN_SERVER_REACHABLE:
    # Set env var to make get_artifact fail-fast without network attempt
    _os.environ["FLASHINFER_NO_DOWNLOAD"] = "1"

    _CUBIN_SKIP_MSG = (
        "Skipped: requires cubin download from "
        "edge.urm.nvidia.com which is unreachable in this environment"
    )
    # Patch 1: trtllm_fp8_block_scale_moe / trtllm_fp8_block_scale_routed_moe (Python-level)
    import flashinfer.fused_moe as _fi_moe_mod
    for _fn_name in ("trtllm_fp8_block_scale_moe", "trtllm_fp8_block_scale_routed_moe"):
        _orig_fn = getattr(_fi_moe_mod, _fn_name, None)
        if _orig_fn is not None:
            def _make_skip_fn(name, orig):
                @_functools.wraps(orig)
                def _skip_fn(*args, **kwargs):
                    pytest.skip(_CUBIN_SKIP_MSG)
                return _skip_fn
            setattr(_fi_moe_mod, _fn_name, _make_skip_fn(_fn_name, _orig_fn))
            import sys as _sys
            for _mod_name, _mod in list(_sys.modules.items()):
                if _mod is not None and hasattr(_mod, _fn_name) and getattr(_mod, _fn_name) is _orig_fn:
                    setattr(_mod, _fn_name, getattr(_fi_moe_mod, _fn_name))

    # Patch 2: cubin_loader.get_artifact -- raises RuntimeError immediately (via FLASHINFER_NO_DOWNLOAD)
    # But since this is called from C ctypes callback, exceptions get swallowed.
    # Patch download_file to raise immediately so the C callback fails fast:
    try:
        import flashinfer.jit.cubin_loader as _cubin_loader_mod
        _orig_download_file = _cubin_loader_mod.download_file
        @_functools.wraps(_orig_download_file)
        def _fast_fail_download_file(source, destination, **kwargs):
            """Fail immediately instead of timing out — server is unreachable."""
            return False
        _cubin_loader_mod.download_file = _fast_fail_download_file
        # Also patch get_artifact for Python-level callers
        _orig_get_artifact = _cubin_loader_mod.get_artifact
        @_functools.wraps(_orig_get_artifact)
        def _skip_get_artifact(file_name, sha256, *args, **kwargs):
            pytest.skip(_CUBIN_SKIP_MSG)
        _cubin_loader_mod.get_artifact = _skip_get_artifact
        if hasattr(_cubin_loader_mod, 'get_cubin'):
            _cubin_loader_mod.get_cubin = _skip_get_artifact
    except Exception as _e:
        pass

# from torch.torch_version import __version__ as torch_version

import flashinfer
# from flashinfer.jit import MissingJITCacheError

# Global tracking for JIT cache coverage
# Store tuples of (test_name, module_name, spec_info)
_MISSING_JIT_CACHE_MODULES: Set[tuple] = set()

# File path for aggregating JIT cache info across multiple pytest runs
_JIT_CACHE_REPORT_FILE = os.environ.get("FLASHINFER_JIT_CACHE_REPORT_FILE", None)

TORCH_COMPILE_FNS = [
    flashinfer.activation.silu_and_mul,
    flashinfer.activation.gelu_and_mul,
    flashinfer.activation.gelu_tanh_and_mul,
    flashinfer.cascade.merge_state,
    flashinfer.cascade.merge_state_in_place,
    flashinfer.cascade.merge_states,
    flashinfer.cascade.MultiLevelCascadeAttentionWrapper.run,
    flashinfer.cascade.BatchDecodeWithSharedPrefixPagedKVCacheWrapper.forward,
    flashinfer.cascade.BatchPrefillWithSharedPrefixPagedKVCacheWrapper.forward,
    flashinfer.decode.single_decode_with_kv_cache,
    flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run,
    flashinfer.gemm.bmm_fp8,
    flashinfer.gemm.SegmentGEMMWrapper.run,
    flashinfer.norm.rmsnorm,
    flashinfer.norm.fused_add_rmsnorm,
    flashinfer.norm.gemma_rmsnorm,
    flashinfer.norm.gemma_fused_add_rmsnorm,
    flashinfer.page.append_paged_kv_cache,
    flashinfer.prefill.single_prefill_with_kv_cache,
    flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run,
    flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper.run,
    flashinfer.quantization.packbits,
    flashinfer.rope.apply_rope,
    flashinfer.rope.apply_rope_inplace,
    flashinfer.rope.apply_rope_pos_ids,
    flashinfer.rope.apply_rope_pos_ids_inplace,
    flashinfer.rope.apply_llama31_rope,
    flashinfer.rope.apply_llama31_rope_inplace,
    flashinfer.rope.apply_llama31_rope_pos_ids,
    flashinfer.rope.apply_llama31_rope_pos_ids_inplace,
    flashinfer.sampling.sampling_from_probs,
    flashinfer.sampling.sampling_from_logits,
    flashinfer.sampling.top_p_sampling_from_probs,
    flashinfer.sampling.top_k_sampling_from_probs,
    flashinfer.sampling.min_p_sampling_from_probs,
    flashinfer.sampling.top_k_top_p_sampling_from_probs,
    flashinfer.sampling.top_p_renorm_probs,
    flashinfer.sampling.top_k_renorm_probs,
    flashinfer.sampling.top_k_mask_logits,
    flashinfer.sampling.chain_speculative_sampling,
]

_TORCH_COMPILE_CACHE: Dict[str, Any] = dict()


def _set_torch_compile_options():
    import torch._dynamo.config

    torch._dynamo.config.cache_size_limit = 128


def _monkeypatch_add_torch_compile(func):
    """
    Replace the given function with its torch.compile version.
    """

    from torch._library.custom_ops import CustomOpDef

    if type(func) is types.FunctionType:
        fn = func
    elif isinstance(func, CustomOpDef):
        fn = func._init_fn
    else:
        raise ValueError(f"Unsupported fn type {type(func)}")

    fullname = fn.__module__ + "." + fn.__qualname__
    components = fullname.split(".")
    assert components[0] == "flashinfer"
    module = flashinfer
    for component in components[1:-1]:
        module = getattr(module, component)
    if not hasattr(module, components[-1]):
        raise ValueError(f"Failed to monkeypatch: {fullname}")

    def wrapper(*args, **kwargs):
        compiled = _TORCH_COMPILE_CACHE.get(fullname)
        if compiled is None:
            # Warmup -- JIT compile / import the kernels.
            #
            # From user side, users also need to warmup the model beforehand,
            # as suggested by PyTorch Cuda Graph docs (not sure if it's also
            # recommended for torch.compile as well.)
            #
            # For the convenience of FlashInfer testing, we do the warmup here,
            # on the first run of the function. The caveat is that the first
            # call will run twice: once to warmup, and another through the
            # compiled version.
            func(*args, **kwargs)

            # Compile
            compiled = torch.compile(
                func,
                fullgraph=True,
                backend="inductor",
                mode="max-autotune-no-cudagraphs",
            )
            _TORCH_COMPILE_CACHE[fn.__name__] = compiled

        return compiled(*args, **kwargs)

    setattr(module, fn.__name__, wrapper)
    print("Applied torch.compile to", fullname)


def pytest_configure(config):
    if os.environ.get("FLASHINFER_TEST_TORCH_COMPILE", "0") == "1":
        # if torch_version < TorchVersion("2.4"):
        #     pytest.skip("torch.compile requires torch >= 2.4")
        _set_torch_compile_options()
        for fn in TORCH_COMPILE_FNS:
            _monkeypatch_add_torch_compile(fn)


def is_cuda_oom_error_str(e: str) -> bool:
    return "CUDA" in e and "out of memory" in e


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # skip OOM error and missing JIT cache errors
    try:
        item.runtest()
    except Exception:
        raise
    # try:
    #     item.runtest()
    # except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    #     if isinstance(e, torch.cuda.OutOfMemoryError) or is_cuda_oom_error_str(str(e)):
    #         pytest.skip("Skipping due to OOM")
    #     elif isinstance(e, MissingJITCacheError):
    #         # Record the test that was skipped due to missing JIT cache
    #         test_name = item.nodeid
    #         spec = e.spec
    #         module_name = spec.name if spec else "unknown"

    #         # Create a dict with module info for reporting
    #         spec_info = None
    #         if spec:
    #             spec_info = {
    #                 "name": spec.name,
    #                 "sources": [str(s) for s in spec.sources],
    #                 "needs_device_linking": spec.needs_device_linking,
    #                 "aot_path": str(spec.aot_path),
    #             }

    #         _MISSING_JIT_CACHE_MODULES.add((test_name, module_name, str(spec_info)))
    #         pytest.skip(f"Skipping due to missing JIT cache for module: {module_name}")
    #     else:
    #         raise


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate JIT cache coverage report at the end of test session"""
    if not _MISSING_JIT_CACHE_MODULES:
        return  # No missing modules

    # If report file is specified, write to file for later aggregation
    # Otherwise, print summary directly
    if _JIT_CACHE_REPORT_FILE:
        from filelock import FileLock

        # Convert set to list for JSON serialization
        data = [
            {"test_name": test_name, "module_name": module_name, "spec_info": spec_info}
            for test_name, module_name, spec_info in _MISSING_JIT_CACHE_MODULES
        ]

        # Use file locking to handle concurrent writes from multiple pytest processes
        Path(_JIT_CACHE_REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
        lock_file = _JIT_CACHE_REPORT_FILE + ".lock"
        with FileLock(lock_file), open(_JIT_CACHE_REPORT_FILE, "a") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return

    # Single pytest run - print summary directly
    terminalreporter.section("flashinfer-jit-cache Package Coverage Report")
    terminalreporter.write_line("")
    terminalreporter.write_line(
        "This report shows the coverage of the flashinfer-jit-cache package."
    )
    terminalreporter.write_line(
        "Tests are skipped when required modules are not found in the installed JIT cache."
    )
    terminalreporter.write_line("")
    terminalreporter.write_line(
        f"⚠️  {len(_MISSING_JIT_CACHE_MODULES)} test(s) skipped due to missing JIT cache modules:"
    )
    terminalreporter.write_line("")

    # Group by module name
    module_to_tests = {}
    for test_name, module_name, spec_info in _MISSING_JIT_CACHE_MODULES:
        if module_name not in module_to_tests:
            module_to_tests[module_name] = {"tests": [], "spec_info": spec_info}
        module_to_tests[module_name]["tests"].append(test_name)

    for module_name in sorted(module_to_tests.keys()):
        info = module_to_tests[module_name]
        terminalreporter.write_line(f"Module: {module_name}")
        terminalreporter.write_line(f"  Spec: {info['spec_info']}")
        terminalreporter.write_line(f"  Affected tests ({len(info['tests'])}):")
        for test in sorted(info["tests"]):
            terminalreporter.write_line(f"    - {test}")
        terminalreporter.write_line("")

    terminalreporter.write_line(
        "These tests require JIT compilation but FLASHINFER_DISABLE_JIT=1 was set."
    )
    terminalreporter.write_line(
        "To improve coverage, add the missing modules to the flashinfer-jit-cache build configuration."
    )
