module OptimizedEinsum

include("Counters.jl")
include("Utils.jl")
include("ContractionPath.jl")
include("Solvers/Solvers.jl")

using Requires: @require
function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("Visualization.jl")
end

export flops, removedsize, rank
export rand_equation, get_symbol
export ContractionPath, subtree, children, inds, draw
import .Solvers: contractpath, Optimal, Greedy, Solver
export contractpath, Optimal, Greedy, Solver

end