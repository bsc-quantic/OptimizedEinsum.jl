# OptimizedEinsum.jl

OptimizedEinsum is a Julia rewrite of the [`opt_einsum`](https://github.com/dgasmith/opt_einsum.git) Python library. It provides routines for Einstein notation ordering and tensor network contraction.

## Alternatives

Unlike other Einsum libraries in Julia, OptimizedEinsum is focused in efficient search of contraction paths for large tensor networks. If you are looking for einsum macros, GPU support or other niceties, we recommend the following alternatives:

- [Einsum](https://github.com/ahwillia/Einsum.jl)
- [OMEinsum](https://github.com/under-Peter/OMEinsum.jl)
- [Tullio.jl](https://github.com/mcabbott/Tullio.jl)
- [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)