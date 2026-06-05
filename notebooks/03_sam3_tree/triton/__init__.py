# Fake triton module for Windows
# SAM3 text-to-mask inference does NOT need real triton.
# Only interactive tracking/EDT needs real triton — and it raises a clear error if called.

import functools

class _JitFunc:
    """Supports fn[grid](args) syntax used by triton kernels."""
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)
    def __getitem__(self, grid):
        return self
    def __call__(self, *args, **kwargs):
        if "debug" in self.fn.__name__ or "init" in self.fn.__name__:
            return  # silently skip init/debug kernels
        raise RuntimeError(
            "triton kernel called but triton is NOT installed on Windows.\n"
            "This only affects interactive tracking/EDT — text-to-mask inference works fine."
        )

def jit(fn, version=None, do_not_use_bf16=False, **kwargs):
    return _JitFunc(fn)

autotune = lambda configs, key, prune_configs_by=None, **kwargs: lambda fn: jit(fn)
heuristics = lambda h, band_width=None: lambda fn: jit(fn)
