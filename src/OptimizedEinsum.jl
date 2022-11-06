module OptimizedEinsum

import Base: show, convert
using Base.Iterators: flatten, repeated, take, drop
using PyCall
using Random

export largest_intermediate
export contract, contract_path
export rand_equation, get_symbol

include("Optimizers/Optimizers.jl")

const oe = PyCall.PyNULL()
const oecontract = PyCall.PyNULL()

function __init__()
    copy!(oe, pyimport_conda("opt_einsum", "opt_einsum"))
    copy!(oecontract, pyimport("opt_einsum.contract"))

    pytype_mapping(oecontract.PathInfo, PathInfo)
end

struct PathInfo
    pyobj::PyObject
end

Base.show(io::IO, p::PathInfo) = print(io, p.pyobj.__repr__())
Base.convert(::Type{PathInfo}, x::PyObject) = PathInfo(x)

largest_intermediate(p::PathInfo) = p.pyobj.largest_intermediate

"""
    contract(subscripts, operands...[, out, dtype, order, casting, use_blas, optimize, memory_limit, backend])

Evaluates the Einstein summation convention on the operands. A drop in
replacement for NumPy's einsum function that optimizes the order of contraction
to reduce overall scaling at the cost of several intermediate arrays.

**Parameters:**

- **subscripts** - *(str)* Specifies the subscripts for summation.
- **\\*operands** - *(list of array_like)* hese are the arrays for the operation.
- **out** - *(array_like)* A output array in which set the sresulting output.
- **dtype** - *(str)* The dtype of the given contraction, see np.einsum.
- **order** - *(str)* The order of the resulting contraction, see np.einsum.
- **casting** - *(str)* The casting procedure for operations of different dtype, see np.einsum.
- **use_blas** - *(bool)* Do you use BLAS for valid operations, may use extra memory for more intermediates.
- **optimize** - *(str, list or bool, optional (default: ``auto``))* Choose the type of path.

    - if a list is given uses this as the path.
    - `'optimal'` An algorithm that explores all possible ways of
      contracting the listed tensors. Scales factorially with the number of
      terms in the contraction.
    - `'dp'` A faster (but essentially optimal) algorithm that uses
      dynamic programming to exhaustively search all contraction paths
      without outer-products.
    - `'greedy'` An cheap algorithm that heuristically chooses the best
      pairwise contraction at each step. Scales linearly in the number of
      terms in the contraction.
    - `'random-greedy'` Run a randomized version of the greedy algorithm
      32 times and pick the best path.
    - `'random-greedy-128'` Run a randomized version of the greedy
      algorithm 128 times and pick the best path.
    - `'branch-all'` An algorithm like optimal but that restricts itself
      to searching 'likely' paths. Still scales factorially.
    - `'branch-2'` An even more restricted version of 'branch-all' that
      only searches the best two options at each step. Scales exponentially
      with the number of terms in the contraction.
    - `'auto'` Choose the best of the above algorithms whilst aiming to
      keep the path finding time below 1ms.
    - `'auto-hq'` Aim for a high quality contraction, choosing the best
      of the above algorithms whilst aiming to keep the path finding time
      below 1sec.

- **memory_limit** - *({None, int, 'max_input'} (default: `None`))* - Give the upper bound of the largest intermediate tensor contract will build.
    - None or -1 means there is no limit
    - `max_input` means the limit is set as largest input tensor
    - a positive integer is taken as an explicit limit on the number of elements

    The default is None. Note that imposing a limit can make contractions
    exponentially slower to perform.

- **backend** - *(str, optional (default: ``auto``))* Which library to use to perform the required ``tensordot``, ``transpose``
    and ``einsum`` calls. Should match the types of arrays supplied, See
    :func:`contract_expression` for generating expressions which convert
    numpy arrays to and from the backend library automatically.

**Returns:**

- **out** - *(array_like)* The result of the einsum expression.

**Notes:**

This function should produce a result identical to that of NumPy's einsum
function. The primary difference is ``contract`` will attempt to form
intermediates which reduce the overall scaling of the given einsum contraction.
By default the worst intermediate formed will be equal to that of the largest
input array. For large einsum expressions with many input arrays this can
provide arbitrarily large (1000 fold+) speed improvements.

For contractions with just two tensors this function will attempt to use
NumPy's built-in BLAS functionality to ensure that the given operation is
performed optimally. When NumPy is linked to a threaded BLAS, potential
speedups are on the order of 20-100 for a six core machine.
"""
function contract(subscripts, operands...; kwargs...)
    oe.contract(subscripts, operands...; kwargs...)
end

"""
    contract_path(subscripts, operands...[, kwargs...])

Find a contraction order `path`, without performing the contraction.

# Arguments
- `subscripts` - *(str)* Specifies the subscripts for summation.
- `operands` - *(list of array_like)* hese are the arrays for the operation.
- `use_blas::Bool`: Do you use BLAS for valid operations, may use extra memory for more intermediates.
- `optimize::Union{String,Vector{Symbol},Bool}`: Choose the type of path.

    - if a list is given uses this as the path.
    - `'optimal'` An algorithm that explores all possible ways of
      contracting the listed tensors. Scales factorially with the number of
      terms in the contraction.
    - `'dp'` A faster (but essentially optimal) algorithm that uses
      dynamic programming to exhaustively search all contraction paths
      without outer-products.
    - `'greedy'` An cheap algorithm that heuristically chooses the best
      pairwise contraction at each step. Scales linearly in the number of
      terms in the contraction.
    - `'random-greedy'` Run a randomized version of the greedy algorithm
      32 times and pick the best path.
    - `'random-greedy-128'` Run a randomized version of the greedy
      algorithm 128 times and pick the best path.
    - `'branch-all'` An algorithm like optimal but that restricts itself
      to searching 'likely' paths. Still scales factorially.
    - `'branch-2'` An even more restricted version of 'branch-all' that
      only searches the best two options at each step. Scales exponentially
      with the number of terms in the contraction.
    - `'auto'` Choose the best of the above algorithms whilst aiming to
      keep the path finding time below 1ms.
    - `'auto-hq'` Aim for a high quality contraction, choosing the best
      of the above algorithms whilst aiming to keep the path finding time
      below 1sec.

- **memory_limit** - *({None, int, 'max_input'} (default: `None`))* - Give the upper bound of the largest intermediate tensor contract will build.

    - None or -1 means there is no limit
    - `max_input` means the limit is set as largest input tensor
    - a positive integer is taken as an explicit limit on the number of elements

    The default is None. Note that imposing a limit can make contractions
    exponentially slower to perform.

- **shapes** - *(bool, optional)* Whether ``contract_path`` should assume arrays (the default) or array shapes have been supplied.

**Returns:**

- **path** - *(list of tuples)* The einsum path
- **PathInfo** - *(str)* A printable object containing various information about the path found.

**Notes:**

The resulting path indicates which terms of the input contraction should be
contracted first, the result of this contraction is then appended to the end of
the contraction list.

**Examples:**

We can begin with a chain dot example. In this case, it is optimal to
contract the b and c tensors represented by the first element of the path (1,
2). The resulting tensor is added to the end of the contraction and the
remaining contraction, `(0, 1)`, is then executed.

```julia
a = rand(2, 2)
b = rand(2, 5)
c = rand(5, 2)
path_info = OptimizedEinsum.contract_path('ij,jk,kl->il', a, b, c)
print(path_info[0])
#> [(1, 2), (0, 1)]
print(path_info[1])
#>   Complete contraction:  ij,jk,kl->il
#>          Naive scaling:  4
#>      Optimized scaling:  3
#>       Naive FLOP count:  1.600e+02
#>   Optimized FLOP count:  5.600e+01
#>    Theoretical speedup:  2.857
#>   Largest intermediate:  4.000e+00 elements
#> -------------------------------------------------------------------------
#> scaling                  current                                remaining
#> -------------------------------------------------------------------------
#>    3                   kl,jk->jl                                ij,jl->il
#>    3                   jl,ij->il                                   il->il
```

A more complex index transformation example.

```julia
I = rand(10, 10, 10, 10)
C = rand(10, 10)
path_info = OptimizedEinsum.contract_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C)

print(path_info[0])
#> [(0, 2), (0, 3), (0, 2), (0, 1)]
print(path_info[1])
#>   Complete contraction:  ea,fb,abcd,gc,hd->efgh
#>          Naive scaling:  8
#>      Optimized scaling:  5
#>       Naive FLOP count:  8.000e+08
#>   Optimized FLOP count:  8.000e+05
#>    Theoretical speedup:  1000.000
#>   Largest intermediate:  1.000e+04 elements
#> --------------------------------------------------------------------------
#> scaling                  current                                remaining
#> --------------------------------------------------------------------------
#>    5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
#>    5               bcde,fb->cdef                         gc,hd,cdef->efgh
#>    5               cdef,gc->defg                            hd,defg->efgh
#>    5               defg,hd->efgh                               efgh->efgh
```
"""
function contract_path(subscripts, operands...; kwargs...)
    oe.contract_path(subscripts, operands...; kwargs...)
end

function contract_path(subscripts, operands::Vararg{<:NTuple{N,Integer} where {N}}; kwargs...)
    oe.contract_path(subscripts, operands...; shapes=true, kwargs...)
end

function contract_path(
    output_inds::NTuple{N,Symbol} where {N},
    operands::Vararg{Pair{NTuple{N,Symbol},NTuple{N,I}} where {N,I<:Integer}};
    kwargs...
)
    oe.contract_path(collect(flatten(reverse.(operands)))..., output_inds; shapes=true, kwargs...)
end

const symbols_base = [Symbol(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]

"""
    get_symbol(i)

Get the symbol corresponding to int ``i`` - runs through the usual 52 letters before resorting to unicode characters, starting at ``Char(192)`` and skipping surrogates.

# Examples
```jldoctest
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

end
