module OptimizedEinsum
using Requires: @require

include("Counters.jl")
include("Utils.jl")
include("ContractionPath.jl")
include("Solvers/Solvers.jl")

function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("Visualization.jl")
    @warn """This is the last release of `OptimizedEinsum` on the General official registry.
        It has been replaced by `EinExprs` (https://github.com/bsc-quantic/EinExprs.jl` which can be found in our registry https://github.com/bsc-quantic/Registry in the form of `OptimizedEinsum` or subdivided packages.
        If you want to use `EinExprs`, add our registry:
        \tusing Pkg
        \tpkg"registry add https://github.com/bsc-quantic/Registry"
        And then,
        \tpkg"add EinExprs"
    """
end

export flops, removedsize, rank
export rand_equation, get_symbol
export ContractionPath, subtree, children, labels
import .Solvers: contractpath, Optimal, Greedy, RandomGreedy, Solver
export contractpath, Optimal, Greedy, RandomGreedy, Solver

end
