module Optimizers

abstract type Optimizer end

function optimize end

optimize(::Type{<:Optimizer}, inputs, output, size_dict) = error("`optimize` not implemented")
optimize(::Optimizer, inputs, output, size_dict) = error("`optimize` not implemented")

include("Optimal.jl")e
include("Greedy.jl")
include("Random.jl")

export Optimizer, optimize
export Optimal, Greedy

end