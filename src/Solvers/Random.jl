abstract type RandomSolver <: Solver end

struct RandomGreedy <: RandomSolver
    temperature::Float32
    rel_temperature::Bool
    nbranch::Int
    cost_fn::Core.Function
end