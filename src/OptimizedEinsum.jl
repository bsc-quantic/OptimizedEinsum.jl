module OptimizedEinsum

include("Counters.jl")
include("Utils.jl")
include("ContractionPath.jl")
include("Visualization.jl")
include("Optimizers/Optimizers.jl")

export flops, removedsize, rank
export rand_equation, get_symbol
export ContractionPath, subtree, children, inds, draw
import .Optimizers: optimize, Optimal, Greedy
export optimize, Optimal, Greedy

end
