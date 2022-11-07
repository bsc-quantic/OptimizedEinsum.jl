using Base: @kwdef

@kwdef struct BranchBound <: Optimizer
    nbranch::Union{Int,Nothing} = Nothing
    cutoff_flops_factor::Int = 4
    minimize::Symbol = :flops
    cost_fn = :memory_removed
end

optimize(::Type{BranchBound}, inputs, output, size) = optimize(BranchBound(), inputs, output, size)

function optimize(config::BranchBound, inputs, output, size)
    error("not yet implemented")
end
