using Base: @kwdef

@kwdef struct DynamicProgramming <: Optimizer
    minimize::Symbol = :flops
    search_outer::Bool = true
    cost_cap::Bool = false
end

optimize(::Type{DynamicProgramming}, inputs, output, size) = optimize(DynamicProgramming(), inputs, output, size)

function optimize(config::DynamicProgramming, inputs, output, size)
    error("not yet implemented")
end