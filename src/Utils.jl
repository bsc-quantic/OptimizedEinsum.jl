using Random
using Base.Iterators: take, drop, repeated

const symbols_base = [Symbol(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]

"""
    get_symbol(i)

Get the symbol corresponding to int ``i`` - runs through the usual 52 letters before resorting to unicode characters, starting at ``Char(192)`` and skipping surrogates.

# Examples
```jldoctest; setup = :(using OptimizedEinsum: get_symbol)
julia> get_symbol(2)
:b

julia> get_symbol(200)
:Ŕ

julia> get_symbol(20000)
:京
```
"""
function get_symbol(i::Integer)
    if i < 52
        Symbol(symbols_base[i])
    elseif i ≥ 55296
        Symbol(Char(i + 2048))
    else
        Symbol(Char(i + 140))
    end
end

"""
    rand_equation(n, reg[, n_out, d_min, d_max, seed, global_dim, return_size_dict])

Generate a random contraction and shapes.

# Arguments
- `n::Integer`: Number of array arguments.
- `reg::Integer`: 'Regularity' of the contraction graph. This essentially determines how many indices each tensor shares with others on average.
- `n_out::Integer=0`: Number of output indices (i.e. the number of non-contracted indices).
- `d_min::Integer=2`: Minimum dimension size.
- `d_max::Integer=9`: Maximum dimension size.
- `seed::Some{Integer}=nothing`: If not None, seed numpy's random generator with this.
- global_dim : bool, optional
    Add a global, 'broadcast', dimension to every operand.

# Returns
- `outer_inds::Vector{Symbol}`: Open indices.
- `inputs::Vector{Vector{Symbol}}`: Indices of tensors.
- `size_dict::Dict[Symbol, Integer]`: The dict of index sizes.

# Examples
```jldoctest
julia> outer_inds, inputs, size_dict = rand_equation(10, 4, n_out=5, seed=42)
([:b, :c, :s, :h, :q], [[:n, :j], [:o, :d, :m], [:w, :v, :x, :i, :f, :u, :h], [:t, :e, :r, :s, :k, :b, :l, :a, :f], [:w, :y], [:r, :g, :p, :n, :y], [:x, :t, :g, :j, :v], [:e, :m, :d], [:p, :i, :a, :c], [:q, :k, :u, :l, :o]], Dict(:o => 7, :b => 6, :p => 6, :n => 6, :j => 4, :e => 5, :c => 4, :h => 7, :l => 7, :w => 6…))

julia> join((join(String(c) for c in str) for str in [inputs..., outer_inds]), ",", "->")
"nj,odm,wvxifuh,terskblaf,wy,rgpny,xtgjv,emd,piac,qkulo->bcshq"
```
---
"""
function rand_equation(n, reg; n_out=0, d_min=2, d_max=9, seed=nothing, global_dim::Bool=false)
    if seed != nothing
        Random.seed!(seed)
    end

    inds = Symbol.(get_symbol.(randperm(n * reg ÷ 2 + n_out)))
    size_dict = Dict(ind => rand(d_min:d_max) for ind in inds)

    outer_inds = take(inds, n_out) |> collect
    inner_inds = drop(inds, n_out) |> collect

    candidate_inds = [outer_inds, flatten(repeated(inner_inds, 2))] |> flatten |> collect |> shuffle

    inputs = map(x -> [x], take(candidate_inds, n))

    for ind in drop(candidate_inds, n)
        i = rand(1:n)
        while ind in inputs[i]
            i = rand(1:n)
        end

        push!(inputs[i], ind)
    end

    if global_dim
        ninds = length(size_dict)
        global_ind = get_symbol(ninds + 1)
        size_dict[global_ind] = rand(d_min:d_max)
        push!(outer_inds, global_ind)
        for input in inputs
            push!(input, global_ind)
        end
    end

    outer_inds, inputs, size_dict
end

"""
    ssa_to_linear(ssa_path)

Convert a path with static single assignment ids to a path with recycled linear ids.

```jldoctest; setup = :(using OptimizedEinsum: ssa_to_linear)
julia> ssa_to_linear([(1,4), (3,5), (2,6)])
3-element Vector{Vector{Int64}}:
 [1, 4]
 [2, 3]
 [1, 2]
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

```jldoctest; setup = :(using OptimizedEinsum: linear_to_ssa)
julia> linear_to_ssa([[1,4], [2,3], [1,2]])
3-element Vector{Vector{Int64}}:
 [1, 4]
 [3, 5]
 [2, 6]
```
"""
function linear_to_ssa(path)
    n = mapreduce(length, +, path) - length(path) + 1
    linear_to_ssa = collect(1:n)
    new_ids = Iterators.countfrom(n + 1)

    ssa_path = map(zip(new_ids, path)) do (new_id, ids)
        ret = map(id -> linear_to_ssa[id], ids) |> collect

        for id in sort(collect(ids); rev=true)
            deleteat!(linear_to_ssa, id)
        end
        append!(linear_to_ssa, new_id)

        ret
    end

    return ssa_path
end

"""
    ssa_path_cost(ssa_path, inputs, output, size)

Compute the flops and max size of an ssa path.
"""
function ssa_path_cost(ssa_path, inputs, output, size)
    inputs = copy(inputs)
    cost = zero(BigInt)
    max_size = one(BigInt)

    for (i, j) in ssa_path
        a, b = inputs[i], inputs[j]
        inds_ij = symdiff(a, b) ∪ ∩(output, a, b)
        flops_ij = flops(a, b, size, output)
        append!(inputs, [inds_ij])

        cost += flops_ij
        max_size = max(max_size, mapreduce(ind -> size[ind], *, inds_ij; init=1))
    end

    return cost, max_size
end

function pathtype(path)
    already_seen = Set{Int}()

    for (a, b) ∈ path
        if a ∈ already_seen || b ∈ already_seen
            return :linear
        end

        push!(already_seen, a)
        push!(already_seen, b)
    end

    return :ssa
end

nonunique(itr...) = nonunique(itr)
nonunique(itr) = nonunique(collect(itr))
function nonunique(itr::Vector)
    xs = sort(itr; by=e -> e isa Tuple || e isa AbstractSet ? collect(e) : e)
    return Set(a for (a, b) ∈ zip(xs, xs[2:end]) if a == b)
end

"""
    popvalue!(d, v)

Like `pop!` but searchs by value and returns the key.
"""
function popvalue!(d, v)
    for (k, vi) in d
        if vi == v
            delete!(d, k)
            return k
        end
    end

    return nothing
end

