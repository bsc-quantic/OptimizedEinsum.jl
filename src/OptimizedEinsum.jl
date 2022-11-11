module OptimizedEinsum

include("Counters.jl")
include("Utils.jl")
include("Optimizers/Optimizers.jl")

export rand_equation, get_symbol

import .Optimizers: optimize, Optimal, Greedy
export optimize, Optimal, Greedy

end
