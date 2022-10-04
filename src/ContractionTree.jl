import AbstractTrees: children
import Base: length, size, getindex
using Base.Iterators: takewhile
using NamedDims

export ContractionTree
export inds, innerinds, outerinds, leaves

"""
	inds(x)

# Notes

```julia
inds(x) == union(innerinds(x), outerinds(x))
````
"""
function inds end

"""
	innerinds(x)
"""
function innerinds end

"""
	outerinds(x)
"""
function outerinds end

"""
    Contraction

Representation of a pairwise contraction operation between tensors.

# Notes
- If only contractions (inner-products), then field `inds` is redundant but we keep it so it can express other operations in the future such as traces, outer/tensor products, element-wise multiplication, ...

- `Contraction.size` should point to the same object as the parent `ContractionTree.size` object.

# To do
- Support other contraction patterns appart of inner-/outer-product
"""
struct Contraction
    inds::Vector{Symbol}
    children::Vector{Contraction}

    # Reference to parent `ContractionTree.size`
    size::Dict{Symbol,Int}

    function Contraction(inds, children)
        # TODO when more contraction patterns are supported, appart of inner-/outer-product, use `union` instead of `symdiff`
        @boundscheck if !issubset(inds, symdiff(inds.(children)...))
            throw(ArgumentError("$inds is not a subset of the indices of its children"))
        end

        new(inds, children)
    end
end

# constructor for default inds (inner product)
function Contraction(children...)
    inds = symdiff(inds.(children)...)

    new(inds, children)
end

children(c::Contraction) = c.children

# TODO check that `inds == outerinds` if just inner-product
inds(c::Contraction) = c.inds
innerinds(c::Contraction) = intersect(inds.(children(c))...)
outerinds(c::Contraction) = symdiff(inds.(children(c))...)

Base.length(c::Contraction) = prod(map(ind -> c.size[ind], inds(c)))
Base.size(c::Contraction) = map(ind -> c.size[ind], inds(c)) |> collect

"""
    ContractionTree

Contraction path of a tensor network represented as a tree.

# Notes
Indices of input tensors are located on the beginning of the `path` field, and the last element corresponds to the root of the tree.
"""
struct ContractionTree
    path::Vector{Contraction}
    size::Dict{Symbol,Int}
end


inds(t::ContractionTree) = union(inds.(t.path)...)
innerinds(t::ContractionTree) = union(innerinds.(t.path)...)
outerinds(t::ContractionTree) = symdiff(outerinds.(t.path)...) # NOTE `outerinds(last(t.path))` might be faster if we keep the root in the end

"""
    root(tree)

Return top `Contraction` of the tree.

# Notes
Current implementation of the `root` method relies in the root being located in the end of `ContractionTree.path`.
"""
root(t::ContractionTree) = last(t.path)

"""
    root(tree)

Return initial `Contraction`s of the tree.

# Notes
Current implementation of the `leaves` method relies in the leaves being located in the beginning of `ContractionTree.path`.
"""
leaves(t::ContractionTree) = takewhile(x -> isempty(children(x)), t.path)

"""
    treewidth(tree)

Return order of the initial or intermediate tensor with maximum order (i.e. largest amount of indices).
"""
treewidth(t::ContractionTree) = maximum(length ∘ outerinds, t.path)

Base.size(t::ContractionTree, ind::Symbol) = t.size[ind]

function subtree(tree::ContractionTree, node::Contraction)
    @assert node ∈ tree.path

    path = [node]
    i = 1
    while checkbounds(path, i)
        append!(path, children(path[i]))
        i += 1
    end

    return ContractionTree(path, t.size)
end

isbinary(t::ContractionTree) = all((length ∘ children)(node) <= 2 for node in t.path)


