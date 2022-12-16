using Base.Iterators: flatten
import Base: maximum, length, IteratorSize

struct ContractionPath
    ssa_path::Vector{NTuple{2,Int}}
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

Base.summary(io::IO, path::ContractionPath) = print(io, "ContractionPath(output=$(isempty(path.output) ? "[]" : path.output), flops=$(flops(path)))")

function signatures(path::ContractionPath)
    inputs = copy(path.inputs)
    for (i, j) ∈ path.ssa_path
        push!(inputs, symdiff(inputs[i], inputs[j]) ∪ ∩(path.output, inputs[i], inputs[j]))
    end

    inputs
end

function inds(path::ContractionPath, i)
    n = length(path.inputs)

    if 1 <= i <= n
        return path.inputs[i]
    end

    (l, r) = path.ssa_path[i-n]
    a = inds(path, l)
    b = inds(path, r)

    return symdiff(a, b) ∪ ∩(path.output, a, b)
end

Base.size(path::ContractionPath, i) = prod(ind -> path.size[ind], inds(path, i), init=1)

IteratorSize(::ContractionPath) = Base.HasLength()

length(path::ContractionPath) = length(path.ssa_path) - 1

Base.iterate(path::ContractionPath, state=0) =
    if state < length(path.ssa_path)
        state += 1
        (path.ssa_path[state], state)
    else
        nothing
    end

function flops(path::ContractionPath)
    signs = signatures(path)
    mapreduce(+, path.ssa_path) do (i, j)
        a = signs[i]
        b = signs[j]
        flops(a, b, path.size, path.output)
    end
end

function flops(path::ContractionPath, i)
    n = length(path.inputs)
    if 1 <= i <= n
        return 0.0
    end

    (l, r) = path.ssa_path[i-n]
    a = inds(path, l)
    b = inds(path, r)

    return flops(a, b, path.size, path.output)
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

children(path::ContractionPath, i) = (n = length(path.inputs); i > n ? path.ssa_path[i-n] : Int[])

function subtree(path::ContractionPath, i)
    if i <= length(path.inputs)
        return nothing
    end

    # select subnodes
    queue = [i]
    for n in queue
        kids = children(path, n)
        if !isempty(kids)
            append!(queue, kids)
        end
    end

    # get final signature and subinds
    signatures = Dict(n => path.inputs[n] for n in Iterators.filter(<=(length(path.inputs)), queue))

    sort!(queue)
    for n in Iterators.filter(>(length(path.inputs)), queue)
        (x, y) = children(path, n)
        a = signatures[x]
        b = signatures[y]

        sign = symdiff(a, b) ∪ ∩(path.output, a, b)

        push!(signatures, n => sign)
    end

    inds = flatten(values(signatures)) |> collect
    size = filter(x -> ((k, _) = x; k ∈ inds), path.size)

    output = signatures[i]
    inputs = [path.inputs[i] for i in queue if i <= length(path.inputs)]

    # translate SSA path
    mapping = Dict{Int,Int}(i => j for (i, j) in zip(queue, Iterators.countfrom(1)))

    ssa_path = [map(x -> mapping[x], path.ssa_path[i-length(path.inputs)])
                for i in queue if i > length(path.inputs)]

    ContractionPath(ssa_path, inputs, output, size)
end
