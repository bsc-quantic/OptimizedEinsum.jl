abstract type RandomOptimizer <: Optimizer end

struct RandomGreedy <: RandomOptimizer
    temperature::Float32
    rel_temperature::Bool
    nbranch::Int
    cost_fn::Core.Function
end