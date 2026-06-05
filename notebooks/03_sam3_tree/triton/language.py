"""Fake triton.language module for Windows."""

import torch

class constexpr:
    """Marks a function parameter as a compile-time constant."""
    def __init__(self, value):
        self.value = value

def program_id(axis=0):
    """Fake: always returns program id 0."""
    return 0

def num_programs(axis=0):
    """Fake: always returns 1 program."""
    return 1

def load(ptr, mask=None, other=None, boundary_check=None, padding_option=None, cache_modifier=None):
    """Fake load: returns 0."""
    if mask is not None and not mask:
        return other if other is not None else 0.0
    return 0.0

def store(ptr, val, mask=None, boundary_check=None, cache_modifier=None):
    """Fake store: no-op."""
    pass

def zeros(shape, dtype=torch.float32):
    """Fake: raises clear error."""
    raise RuntimeError("triton.language.zeros called — triton not available")

def full(shape, value, dtype=torch.float32):
    raise RuntimeError("triton.language.full called — triton not available")

def arange(start, end, step=1):
    raise RuntimeError("triton.language.arange called — triton not available")

def static_range(start, end=None, step=1):
    """Fake iterator: yields nothing."""
    return iter(())

def max(x, axis=None):
    return x

def min(x, axis=None):
    return x

def sum(x, axis=None):
    return x

def abs(x):
    return x.abs() if hasattr(x, 'abs') else x

def sqrt(x):
    return x.sqrt() if hasattr(x, 'sqrt') else x

def where(condition, x, y):
    return x if condition else y

def cat(xs, axis=0):
    return xs[0]

float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
int32 = torch.int32
uint32 = torch.uint32

class Tensor:
    """Stub tensor wrapper."""
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data

def cdiv(x, y):
    """Ceiling division."""
    return (x + y - 1) // y

def ravel(x):
    return x
