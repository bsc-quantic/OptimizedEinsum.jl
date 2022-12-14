using Base: @kwdef

@kwdef struct DynamicProgramming <: Solver
    minimize::Symbol = :flops
    search_outer::Bool = true
    cost_cap::Bool = false
end

function optimize(config::DynamicProgramming, inputs, output, size)
    error("not yet implemented")
end