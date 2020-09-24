import numpy as np


def shape_assert(array,abstract_shape):
    assert len(array.shape)==len(abstract_shape),(array.shape,abstract_shape)
    for (a,b) in zip(array.shape,abstract_shape):
        if b is not None:
            assert a==b,(array.shape,abstract_shape)

def _np(array_or_tensor):
    if type(array_or_tensor) is np.ndarray:
        return array_or_tensor
    else:
        return array_or_tensor.numpy()

def unbatch(a):
    if a.shape[0]==1:
        return a[0]
    else:
        raise ValueError("no batch dim or batch size > 1")

def assert_same_len(*args):
    l = len(args[0])
    assert all(len(a)==l for a in args),[len(a) for a in args]
