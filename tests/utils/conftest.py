# Paddle compat patches for tests/utils/
#
# §44: assert_close bfloat16/float16 — paddle.isclose not registered
# §45: outer RuntimeError wraps inner; walk __cause__/__context__ chain
# §46: torch.equal returns element-wise bool Tensor; use .all().item()
# §47: tensor.multiply(scalar_int) — Paddle multiply requires both Tensors
# §48: tensor.clamp_min/clamp_max — PyTorch aliases missing on Paddle
# §49: torch.sort — axis vs dim + returns only values not (values,indices)
# §50: torch.randn/rand(generator=) — Paddle does not support generator
import functools
import torch
from collections import namedtuple as _namedtuple


# -- §44/§45: assert_close bfloat16/float16 fix --------------------------

_orig_assert_close = torch.testing.assert_close


def _is_paddle_isclose_dtype_error(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur)
        if ('bfloat16' in msg or 'float16' in msg) and (
            'isclose' in msg or 'NotFound' in msg
        ):
            return True
        cur = getattr(cur, '__cause__', None) or getattr(cur, '__context__', None)
    return False


def _manual_allclose(actual, expected, rtol, atol):
    import numpy as np
    a = actual.float().detach().cpu().numpy()
    e = expected.float().detach().cpu().numpy()
    diff = np.abs(a - e)
    tol = atol + rtol * np.abs(e)
    if not np.all(diff <= tol):
        max_diff = float(diff.max())
        max_loc = np.unravel_index(diff.argmax(), diff.shape)
        raise AssertionError(
            'Tensors are not close! '
            'Max diff: ' + str(max_diff) + ' at ' + str(max_loc) + ' '
            'rtol=' + str(rtol) + ' atol=' + str(atol)
        )


@functools.wraps(_orig_assert_close)
def _paddle_compat_assert_close(actual, expected, *args, **kwargs):
    try:
        _orig_assert_close(actual, expected, *args, **kwargs)
    except RuntimeError as e:
        if _is_paddle_isclose_dtype_error(e):
            rtol = kwargs.get('rtol', None)
            atol = kwargs.get('atol', None)
            dt = actual.dtype if isinstance(actual, torch.Tensor) else torch.float32
            if rtol is None:
                rtol = 0.016 if dt == torch.bfloat16 else (0.001 if dt == torch.float16 else 1.3e-6)
            if atol is None:
                atol = 1e-5
            _manual_allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            raise


torch.testing.assert_close = _paddle_compat_assert_close


# -- §46: torch.equal returns element-wise Tensor not bool scalar ---------

_orig_equal = torch.equal


@functools.wraps(_orig_equal)
def _paddle_compat_equal(input, other):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        if input.shape != other.shape:
            return False
    result = _orig_equal(input, other)
    if isinstance(result, torch.Tensor):
        if result.numel() == 1:
            return bool(result.item())
        else:
            return bool(result.all().item())
    return bool(result)


torch.equal = _paddle_compat_equal


# -- §47: tensor.multiply(scalar) / torch.multiply(tensor, scalar) --------

_orig_tensor_multiply = torch.Tensor.multiply


def _paddle_compat_tensor_multiply(self, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    return _orig_tensor_multiply(self, other)


torch.Tensor.multiply = _paddle_compat_tensor_multiply

_orig_torch_multiply = torch.multiply


@functools.wraps(_orig_torch_multiply)
def _paddle_compat_torch_multiply(input, other, **kwargs):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    elif isinstance(input, (int, float)):
        input = torch.tensor(input, dtype=other.dtype, device=other.device)
    return _orig_torch_multiply(input, other, **kwargs)


torch.multiply = _paddle_compat_torch_multiply

_orig_tensor_multiply_ = torch.Tensor.multiply_


def _paddle_compat_tensor_multiply_(self, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    return _orig_tensor_multiply_(self, other)


torch.Tensor.multiply_ = _paddle_compat_tensor_multiply_

# -- §51: tensor.mul(scalar) — Paddle mul requires both Tensors ----------

_orig_tensor_mul = torch.Tensor.mul


def _paddle_compat_tensor_mul(self, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    return _orig_tensor_mul(self, other)


torch.Tensor.mul = _paddle_compat_tensor_mul

_orig_torch_mul = torch.mul


def _paddle_compat_torch_mul(input, other, **kwargs):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    elif isinstance(input, (int, float)):
        input = torch.tensor(input, dtype=other.dtype, device=other.device)
    return _orig_torch_mul(input, other, **kwargs)


torch.mul = _paddle_compat_torch_mul


# -- §48: tensor.clamp_min/clamp_max missing on Paddle --------------------

def _clamp_min(self, min_val):
    return torch.clamp(self, min=min_val)


def _clamp_max(self, max_val):
    return torch.clamp(self, max=max_val)


torch.Tensor.clamp_min = _clamp_min
torch.Tensor.clamp_max = _clamp_max


# -- §49: torch.sort axis/dim + (values,indices) return ------------------
# Paddle compat: uses axis= not dim=; returns bare Tensor not (vals,idxs)

_SortResult = _namedtuple('sort', ['values', 'indices'])
_orig_torch_sort = torch.sort
_orig_torch_argsort = torch.argsort


def _make_sort_result(input_tensor, result, axis, descending, stable):
    if not isinstance(result, torch.Tensor):
        return result
    try:
        indices = _orig_torch_argsort(input_tensor, axis, descending=descending, stable=stable)
    except Exception:
        try:
            indices = _orig_torch_argsort(input_tensor, axis, descending=descending)
        except Exception:
            indices = _orig_torch_argsort(input_tensor, axis)
    return _SortResult(values=result, indices=indices)


@functools.wraps(_orig_torch_sort)
def _paddle_compat_sort(input, *args, **kwargs):
    if 'dim' in kwargs:
        kwargs['axis'] = kwargs.pop('dim')
    axis = kwargs.get('axis', args[0] if args else -1)
    descending = kwargs.get('descending', False)
    stable = kwargs.get('stable', False)
    result = _orig_torch_sort(input, *args, **kwargs)
    return _make_sort_result(input, result, axis, descending, stable)


torch.sort = _paddle_compat_sort

_orig_tensor_sort = torch.Tensor.sort


def _paddle_compat_tensor_sort(self, *args, **kwargs):
    if 'dim' in kwargs:
        kwargs['axis'] = kwargs.pop('dim')
    axis = kwargs.get('axis', args[0] if args else -1)
    descending = kwargs.get('descending', False)
    stable = kwargs.get('stable', False)
    result = _orig_tensor_sort(self, *args, **kwargs)
    return _make_sort_result(self, result, axis, descending, stable)


torch.Tensor.sort = _paddle_compat_tensor_sort


# -- §50: torch.randn/rand(generator=) not supported by Paddle -----------

_orig_torch_randn = torch.randn


@functools.wraps(_orig_torch_randn)
def _paddle_compat_randn(*args, **kwargs):
    kwargs.pop('generator', None)
    return _orig_torch_randn(*args, **kwargs)


torch.randn = _paddle_compat_randn

_orig_torch_rand = torch.rand


@functools.wraps(_orig_torch_rand)
def _paddle_compat_rand(*args, **kwargs):
    kwargs.pop('generator', None)
    return _orig_torch_rand(*args, **kwargs)


torch.rand = _paddle_compat_rand


# §52: torch.cuda.init() — paddle.cuda has no init(); patch as no-op (§52)
if not hasattr(torch.cuda, 'init') or callable(getattr(torch.cuda, 'init', None)) is False:
    def _paddle_compat_cuda_init():
        pass
    torch.cuda.init = _paddle_compat_cuda_init
else:
    _orig_cuda_init = torch.cuda.init
    def _paddle_compat_cuda_init():
        try:
            _orig_cuda_init()
        except Exception:
            pass
    torch.cuda.init = _paddle_compat_cuda_init
