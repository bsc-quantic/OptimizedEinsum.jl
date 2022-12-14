module OptimizedEinsum

include("Counters.jl")
include("Utils.jl")
include("ContractionPath.jl")
include("Visualization.jl")
include("Solvers/Solvers.jl")

export flops, removedsize, rank
export rand_equation, get_symbol
export ContractionPath, subtree, children, inds, draw
import .Solvers: contractpath, Optimal, Greedy
export contractpath, Optimal, Greedy

end
