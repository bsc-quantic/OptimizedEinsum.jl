using Graphs
using Makie
using GraphMakie

const DEFAULT_LEAF_NODE_SIZE = 5.0

Makie.plottype(::ContractionPath) = GraphPlot

function Makie.plot!(P::GraphPlot{Tuple{ContractionPath}}; kwargs...)
    path = P[1][]
    graph = SimpleDiGraph([Edge(from, to) for (pair, to) in zip(path.ssa_path, Iterators.countfrom(length(path.inputs) + 1)) for from in pair])

    kwargs = Dict{Symbol,Any}(kwargs)
    scene = Makie.parent_scene(P)
    default_attrs = default_theme(scene, GraphPlot)

    if P.attributes.edge_width[] == default_attrs.edge_width[]
        kwargs[:edge_width] = [log10(size(path, i)) for i in 1:ne(graph)]
    end

    if P.attributes.arrow_size[] == default_attrs.arrow_size[]
        kwargs[:arrow_size] = [3 * log10(size(path, i)) for i in 1:ne(graph)]
    end

    if P.attributes.edge_color[] == default_attrs.edge_color[]
        kwargs[:edge_color] = [log10(size(path, i)) for i in 1:ne(graph)]
    end

    if P.attributes.node_size[] == default_attrs.node_size[]
        kwargs[:node_size] = [(x = log2(flops(path, i)); isinf(x) ? DEFAULT_LEAF_NODE_SIZE : x) for i in 1:nv(graph)]
    end

    graphplot!(P, graph; kwargs...)
end