module Solvers

abstract type Solver end

function contractpath end

contractpath(T::Type{<:Solver}, inputs, output, size_dict, kwargs...) = contractpath(T(kwargs...), inputs, output, size_dict)

include("Optimal.jl")
include("Greedy.jl")
include("DynamicProgramming.jl")
include("Branch.jl")
include("Random.jl")

export Solver, contractpath
export Optimal, Greedy, RandomGreedy

end