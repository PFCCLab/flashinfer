# Paddle compat patches for tests/model_optimizations/
# Reuses patterns from tests/utils/conftest.py plus additional fixes.
#
# §44/§45: assert_close bfloat16/float16 — paddle.isclose not registered
# §45: Walk __cause__/__context__ to find inner paddle error
# §46: torch.equal() returns Tensor not bool
# §47: tensor.multiply(scalar) — Paddle requires Tensor args
# §48: tensor.clamp_min/clamp_max — missing on Paddle Tensors
# §51: torch.sort() in Paddle compat returns only values (no indices);
#      also does not accept dim= keyword argument (uses axis=).
#      Fix: monkey-patch to accept dim= kwarg and return proper (values, indices) tuple.
import functools
import torch


# ── §44/§45: assert_close bfloat16/float16 fix ──────────────────────────────

_orig_assert_close = torch.testing.assert_close


def _is_paddle_isclose_dtype_error(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur)
        if ("bfloat16" in msg or "float16" in msg) and (
            "isclose" in msg or "NotFound" in msg
        ):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
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
            f"Tensors are not close!\n"
            f"Max absolute diff: {max_diff:.6f} at {max_loc}\n"
            f"actual[{max_loc}]={float(a[max_loc]):.6f}, "
            f"expected[{max_loc}]={float(e[max_loc]):.6f}\n"
            f"rtol={rtol}, atol={atol}"
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


# ── §46: torch.equal returns Tensor instead of bool ─────────────────────────

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


# ── §47: tensor.multiply(scalar) ─────────────────────────────────────────────

_orig_tensor_multiply = torch.Tensor.multiply


def _paddle_compat_tensor_multiply(self, other):
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=self.dtype, device=self.device)
    return _orig_tensor_multiply(self, other)


torch.Tensor.multiply = _paddle_compat_tensor_multiply


# ── §48: clamp_min / clamp_max ───────────────────────────────────────────────

torch.Tensor.clamp_min = lambda self, v: torch.clamp(self, min=v)
torch.Tensor.clamp_max = lambda self, v: torch.clamp(self, max=v)


# ── §51: torch.sort — dim= kwarg unsupported; returns values-only Tensor ─────
# Paddle's sort uses axis= not dim=, and returns only the sorted values tensor.
# Fix: accept dim= kwarg (pass as positional), and wrap result to return
# a proper (values, indices) pair when Paddle returns only values.

_orig_sort = torch.sort


class _SortResult:
    """Mimics torch.return_types.sort for Paddle compat."""

    __slots__ = ("values", "indices")

    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

    def __iter__(self):
        return iter((self.values, self.indices))

    def __getitem__(self, i):
        return (self.values, self.indices)[i]

    def __len__(self):
        return 2


def _paddle_compat_sort(input, dim=-1, descending=False, stable=False, *, out=None):
    # Paddle sort does not accept dim= as keyword; pass as positional.
    # Also ignores stable= (Paddle sort is always stable).
    try:
        result = _orig_sort(input, dim, descending=descending)
    except TypeError:
        # Fallback: try without dim (1-D case)
        result = _orig_sort(input, descending=descending)

    if isinstance(result, torch.Tensor):
        # Paddle returned values-only tensor; compute indices via argsort.
        # Use stable=True to match PyTorch's default stable sort behavior.
        try:
            indices = torch.argsort(input, dim, descending=descending, stable=True)
        except TypeError:
            indices = torch.argsort(input, descending=descending, stable=True)
        return _SortResult(result, indices)

    # Already a proper (values, indices) pair
    return result


torch.sort = _paddle_compat_sort
