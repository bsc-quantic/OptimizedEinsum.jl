using Base: @kwdef

@kwdef struct BranchBound <: Solver
    nbranch::Union{Int,Nothing} = Nothing
    cutoff_flops_factor::Int = 4
    minimize::Symbol = :flops
    cost_fn = :memory_removed
end

function optimize(config::BranchBound, inputs, output, size)
    error("not yet implemented")
end
