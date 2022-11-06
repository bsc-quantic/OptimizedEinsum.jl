module Optimizers

abstract type Optimizer end

function optimize end

optimize(::Type{<:Optimizer}, inputs, output, size_dict) = error("`optimize` not implemented")
optimize(::Optimizer, inputs, output, size_dict) = error("`optimize` not implemented")

include("Optimal.jl")
include("Greedy.jl")
include("Random.jl")

export Optimizer, optimize
export Optimal, Greedy

"""
    ssa_to_linear(ssa_path)

Convert a path with static single assignment ids to a path with recycled linear ids.

```jldoctest
julia> ssa_to_linear([(1,4), (3,5), (2,6)])
3-element Vector{Tuple{Int64, Int64}}:
 (1, 4)
 (2, 3)
 (1, 2)
```
"""
function ssa_to_linear(ssa_path)
    ids = 1:maximum(Iterators.flatten(ssa_path)) |> collect

    path = map(ssa_path) do ssa_ids
        ret = map(ssa_id -> ids[ssa_id], ssa_ids) |> collect

        for ssa_id in ssa_ids
            ids[ssa_id:end] .-= 1
        end

        ret
    end |> collect

    return path
end

"""
    linear_to_ssa(path)

Convert a path with recycled linear ids to a path with static single assignment ids.

```jldoctest
julia> linear_to_ssa([[1,4], [2,3], [1,2]])
3-element Vector{Tuple{Int64, Int64}}:
 (1, 4)
 (3, 5)
 (2, 6)
```
"""
function linear_to_ssa(path::Vector{Vector{Int}})
    n = mapreduce(length, +, path) - length(path) + 1
    linear_to_ssa = collect(1:n)
    new_ids = Iterators.countfrom(n + 1)

    ssa_path = map(zip(new_ids, path)) do (new_id, ids)
        ret = map(id -> linear_to_ssa[id], ids) |> collect

        for id in sort(ids, rev=true)
            deleteat!(linear_to_ssa, id)
        end
        append!(linear_to_ssa, new_id)

        ret
    end

    return ssa_path
end

end