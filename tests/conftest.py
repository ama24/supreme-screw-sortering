import os

# One BLAS thread: OpenBLAS otherwise reserves buffers per core, which can fail when Windows'
# commit memory is nearly exhausted (the test process then exits silently).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
