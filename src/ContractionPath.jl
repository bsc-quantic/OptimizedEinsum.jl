using Base.Iterators: flatten
import Base: maximum

struct ContractionPath
    ssa_path::Vector{Vector{Int}}
    inputs::Vector{Vector{Symbol}}
    output::Vector{Symbol}
    size::Dict{Symbol,Int}

    function ContractionPath(path, inputs, output, size)
        @assert issetequal(flatten([inputs..., output]), keys(size))

        if pathtype(path) == :linear
            path = linear_to_ssa(path)
        end

        new(path, inputs, output, size)
    end
end

function signatures(path::ContractionPath)
    inputs = copy(path.inputs)
    for (i, j) ∈ path.ssa_path
        push!(inputs, symdiff(inputs[i], inputs[j]) ∪ ∩(path.output, inputs[i], inputs[j]))
    end

    inputs
end

function flops(path::ContractionPath)
    signs = signatures(path)
    mapreduce(+, path.ssa_path) do (i, j)
        a = signs[i]
        b = signs[j]
        flops(a, b, path.size, path.output)
    end
end

function maximum(fn, path::ContractionPath)
    signs = signatures(path)
    mapreduce(max, path.ssa_path) do (i, j)
        a = signs[i]
        b = signs[j]
        fn(a, b, path.size, path.output)
    end
end

maximum(path::ContractionPath) = maximum(flops, path)