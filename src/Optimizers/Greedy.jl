using DataStructures: BinaryMinHeap
using Base: @kwdef
using OptimizedEinsum: removedsize, ssa_to_linear, nonunique, ContractionPath
using Combinatorics: combinations

"""
Greedy contraction path solver.

1. Eagerly compute Hadamard products.
2. Greedily compute contractions to maximize `removed_size`.
3. Greedily compute outer products.
"""
@kwdef struct Greedy <: Optimizer
    choose_fn::Function = greedy_choose_simple!
    cost_fn::Function = removedsize
end

function optimize(::Type{Greedy}, inputs, output, size, kwargs...)
    config = Greedy(kwargs...)
    optimize(config, inputs, output, size)
end

function optimize(config::Greedy, inputs, output, size)
    # TODO memory limit?
    ssa_path = ssa_greedy_optimize(inputs, output, size, config.choose_fn, config.cost_fn)

    ContractionPath(ssa_path, inputs, output, size)
end

struct HeapNode{C,M}
    cost::C
    meta::M

    function HeapNode(cost::C, meta::M) where {C,M}
        new{C,M}(cost, meta)
    end
end

Base.isequal(a::HeapNode, b::HeapNode) = isequal(a.cost, b.cost)
Base.isless(a::HeapNode, b::HeapNode) = isless(a.cost, b.cost)
meta(node::HeapNode) = node.meta

function ssa_greedy_optimize(inputs, output, size, choose_fn=greedy_choose_simple!, cost_fn=removedsize)
    # trivial case
    if length(inputs) == 1
        return [(0,)]
    end

    # `choose_fn` chooses which contraction candidate to take
    push_all = !(choose_fn == greedy_choose_simple!)

    # only care about indices, not its order
    inputs = map(Set, inputs)

    # indices shared by all tensors can be seen as output indices
    output = output ∪ ∩(inputs...)

    ssa_path = Vector{NTuple{2,Int}}()

    # step 1: deduplicate shapes by eagerly computing Hadamard products (i.e. element-wise multiplication)
    remaining = Dict{Set{Symbol},Int}(inds => i for (i, inds) ∈ enumerate(inputs))
    ssa_ids = length(inputs) + 1

    for input ∈ nonunique(inputs)
        reduce(Iterators.filter(isequal(input) ∘ last, enumerate(inputs))) do (i, _), (j, _)
            push!(ssa_path, (i, j))

            new_ssa_id = ssa_ids
            ssa_ids += 1
            remaining[i] = new_ssa_id
            return (new_ssa_id, nothing)
        end
    end

    # step 2: greedily compute contractions to minimize `cost_fn` (i.e. maximize `removed_size`)

    # list of indices to contract
    target_inds = setdiff(unique(Iterators.flatten(keys(remaining))), output)

    # histogram of ocurrences of target indices
    # i.e. a index can only be contracted if it only appears in 2 tensors
    # if it appears in 3+, then it cannot be contracted (but indirect Hadamard products can)
    target_inds_histogram = histogram(Iterators.filter(∋(target_inds), Iterators.flatten(keys(remaining))))
    high_ocurrent_inds = keys(filter(>(2) ∘ last, target_inds_histogram))

    # generate candidate pairwise contractions
    queue = BinaryMinHeap{HeapNode{Int,NTuple{3,Set{Symbol}}}}()

    for xs ∈ Iterators.map(ind -> filter(∋(ind), keys(remaining)), target_inds)
        for (a, b) ∈ combinations(xs, 2)
            # result label signature
            c = symdiff(a, b) ∪ ∩(output ∪ high_ocurrent_inds, a, b)

            # compute heuristic cost of candidate
            cost = cost_fn(a, b, size, output)

            # add candidate to queue
            push!(queue, HeapNode(cost, (a, b, c)))
        end
    end

    while !isempty(queue)
        # select candidate
        winner = choose_fn(queue, remaining)
        if winner == nothing
            continue
        end
        (inds_i, inds_j, inds_k) = meta(winner)

        # append winner to contraction path
        ssa_id_i = pop!(remaining, inds_i)
        ssa_id_j = pop!(remaining, inds_j)
        push!(ssa_path, (ssa_id_i, ssa_id_j))
        remaining[inds_k] = ssa_ids
        ssa_ids += 1

        # update candidate queue
        neighbours = Set(inds for inds ∈ keys(remaining) if !isdisjoint(inds, inds_k) && inds != inds_k)
        if !isempty(neighbours)
            inds_i = inds_k
            for inds_j ∈ neighbours
                # output inds and inds with an ocurrence higher or equal to 3 cannot be contracted
                # (in the latest, a Hadamard product can be performed)
                high_ocurrent_inds = keys(filter(>(2) ∘ last, target_inds_histogram))
                inds_k = symdiff(inds_i, inds_j) ∪ ∩(output ∪ high_ocurrent_inds, inds_i, inds_j)

                # compute heuristic cost of candidate
                cost = cost_fn(inds_i, inds_j, size, output)

                # add candidate to queue
                push!(queue, HeapNode(cost, (inds_i, inds_j, inds_k)))

                # update ocurrence histogram
                for ind ∈ ∩(target_inds, inds_i, inds_j)
                    target_inds_histogram[ind] -= 1
                end
            end
        end
    end

    # step 3: greedily compute outer products
    queue = MutableBinaryMinHeap([(prod(size[ind] for ind ∈ inds ∩ output), ssa_id, inds) for (inds, ssa_id) ∈ remaining])

    while ((_, ssa_id_i, inds_i) = pop!(queue); !isempty(queue))
        (_, ssa_id_j, inds_j) = pop!(queue)

        ssa_id_i, ssa_id_j = minmax(ssa_id_i, ssa_id_j)
        push!(ssa_path, (ssa_id_i, ssa_id_j))

        inds_k = (inds_i ∪ inds_j) ∩ output
        cost = prod(size[ind] for ind ∈ inds_k)

        push!(queue, (cost, ssa_id_k, inds_k))
    end

    return ssa_path
end

function greedy_choose_simple!(queue, remaining)
    node = pop!(queue)
    inds_i, inds_j, _ = meta(node)

    if any(inds ∉ keys(remaining) for inds ∈ [inds_i, inds_j])
        return nothing
    end

    return node
end

function histogram(itr)
    # NOTE maybe use `Base.IteratorEltype` for using `Any`?
    hist = Dict{eltype(itr),Int}()
    for el in itr
        hist[el] = get(hist, el, 0) + 1
    end

    hist
end
