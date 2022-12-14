using Graphs
using Makie
using GraphMakie

const DEFAULT_LEAF_NODE_SIZE = 5.0

Makie.plottype(::ContractionPath) = GraphPlot

function Makie.plot!(P::GraphPlot{Tuple{ContractionPath}}; kwargs...)
    path = P[1][]
    graph = SimpleDiGraph([Edge(from, to) for (pair, to) in zip(path.ssa_path, Iterators.countfrom(length(path.inputs) + 1)) for from in pair])

    kwargs = Dict{Symbol,Any}(kwargs)
    get!(kwargs, :edge_width) do
        [log10(size(path, i)) for i in 1:ne(graph)]
    end

    get!(kwargs, :arrow_size) do
        [3 * log10(size(path, i)) for i in 1:ne(graph)]
    end

    get!(kwargs, :edge_color) do
        [log10(size(path, i)) for i in 1:ne(graph)]
    end

    get!(kwargs, :node_size) do
        [(x = log2(flops(path, i)); isinf(x) ? DEFAULT_LEAF_NODE_SIZE : x) for i in 1:nv(graph)]
    end

    graphplot!(P, graph; kwargs...)
end