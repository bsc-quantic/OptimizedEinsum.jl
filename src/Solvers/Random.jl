using OptimizedEinsum: removedsize, ssa_path_cost, ContractionPath
using OptimizedEinsum.Solvers: greedy_choose_simple!, greedy_choose_thermal!
using Base: @kwdef
using Distributions: Normal, truncated
using Random: seed!

abstract type RandomSolver <: Solver end

@kwdef struct RandomGreedy <: RandomSolver
    temperature::Float32 = 1.
    rel_temperature::Bool = true
    nbranch::Int = 8
    repeats::Int = 128
    choose_fn::Function = (x, y) -> greedy_choose_thermal!(x, y, nbranch, temperature, rel_temperature)
    cost_fn::Function = (a, b, size, keep) -> rand(truncated(Normal(1.0, 0.01),0,10),1)[1]*removedsize(a, b, size, keep)
    minimize::Symbol = :flops # :flops or :size
end

function contractpath(config::RandomGreedy, inputs, output, size)
    best = Dict{String,Any}("flops" => Inf, "size" => Inf)

    trials = [trail_greedy_ssa_path_and_cost(i, inputs, output, size, config.choose_fn, config.cost_fn) for i in 1:config.repeats]

    # assess the trials
    for (ssa_path, cost, size) in trials

        if is_better(config.minimize, cost, size, best["flops"], best["size"])
            best["flops"] = cost
            best["size"] = size
            best["ssa_path"] = ssa_path
        end
    end

    ContractionPath(best["ssa_path"], inputs, output, size)
end

function trail_greedy_ssa_path_and_cost(r, inputs, output, size_dict, choose_fn, cost_fn)
    seed!(r)

    ssa_path = ssa_greedy_optimize(inputs, output, size_dict, choose_fn, cost_fn)
    flops, size = ssa_path_cost(ssa_path, inputs, output, size_dict)

    return ssa_path, flops, size
end

function is_better(minimize::Symbol, flops::Number, size::Number, best_flops::Number, best_size::Number)
    if minimize == :flops
        return (flops, size) < (best_flops, best_size)
    elseif minimize == :size
        return (size, flops) < (best_size, best_flops)
    else
        error("minimize = $minimize is not implemented")
    end
end