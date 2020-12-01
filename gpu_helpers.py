import pycuda.gpuarray as gp
from pycuda.driver import memcpy_dtod_async

def provide_pycuda_array(ptr, n_entries, dtype=float):
    return gp.GPUArray(n_entries, dtype=dtype, gpudata=ptr)

def gpuarray_memcpy(dest, src):
    '''Device memory copy with pycuda from
    src GPUArray to dest GPUArray.
    '''
#     dest[:] = src
#     memcpy_atoa(dest, 0, src, 0, len(src))
    memcpy_dtod_async(dest.gpudata, src.gpudata, src.nbytes)
