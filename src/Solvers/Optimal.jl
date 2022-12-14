using Combinatorics
using OptimizedEinsum: ssa_to_linear, flops, ContractionPath

@doc raw"""
`Optimal` contraction path solver guarantees to find the optimal contraction path **always**, but at the cost of factorial ``\mathcal{O}(n!)`` time complexity.
"""
struct Optimal <: Solver end

function contractpath(::Optimal, inputs, output, size)
    n = length(inputs)

    best_flops = Ref(typemax(Int128)) # TODO use `BigInt`?
    best_ssa_path = Ref{Vector{NTuple{2,Int}}}()


    function _iterate(path, inputs, remaining, flops_cur)
        # end of path (only reached if flops is best so far)
        if length(remaining) == 1
            best_flops[] = flops_cur
            best_ssa_path[] = path
            return
        end

        # depth-first search of paths
        for (i, j) in combinations(remaining |> collect, 2)
            (i, j) = minmax(i, j)
            candidate = (inputs[i], inputs[j])

            # resulting inds and flops of the pairwise contraction of `i` and `j`
            inds_ij = symdiff(candidate...) ∪ ∩(output, candidate...)
            flops_ij = flops(candidate..., size, output)

            # prune paths based on flops
            flops_candidate = flops_cur + flops_ij
            if flops_candidate >= best_flops[]
                continue
            end

            # TODO prune paths based on memory limit?

            # descend one level fixing `candidate`
            _iterate(
                vcat(path, [(i, j)]), # path
                vcat(inputs, [inds_ij]), # inputs
                setdiff(remaining, i, j) ∪ (length(inputs) + 1), # remaining
                flops_candidate, # flops_cur
            )
        end
    end

    path = Vector{NTuple{2,Int}}()
    _iterate(path, inputs, Set(1:n), 0)

    ContractionPath(best_ssa_path[], inputs, output, size)
end