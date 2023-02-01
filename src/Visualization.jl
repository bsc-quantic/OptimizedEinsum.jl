using OptimizedEinsum
using GraphMakie
using Graphs
using Makie
using LaTeXStrings

const MIN_LOGFLOP = 0.01
const MAX_EDGE_WIDTH = 10.
const MAX_ARROW_SIZE = 28.
const MAX_NODE_SIZE = 40.

function Makie.plot(path::ContractionPath; colormap=to_colormap(:viridis)[begin:end-10], kwargs...)
    graph = SimpleDiGraph([Edge(from, to) for (pair, to) in zip(path.ssa_path, Iterators.countfrom(length(path.inputs) + 1)) for from in pair])

    f, ax, p = graphplot(wheel_graph(1))

    kwargs = Dict{Symbol,Any}(kwargs)
    scene = Makie.parent_scene(p)

    default_attrs = default_theme(scene, GraphPlot)

    log_size = [log2(size(path, i)) for i in 1:ne(graph)]
    log_flop = [(x = log10(flops(path, i)); isinf(x) ? MIN_LOGFLOP : x) for i in 1:nv(graph)]

    min_size, max_size = extrema(log_size)
    min_flop, max_flop = extrema(log_flop)

    if p.attributes.edge_width[] == default_attrs.edge_width[]
        kwargs[:edge_width] = (log_size/max_size)*10
    end

    if p.attributes.arrow_size[] == default_attrs.arrow_size[]
        kwargs[:arrow_size] = (log_size/max_size)*28
    end

    if p.attributes.edge_color[] == default_attrs.edge_color[]
        kwargs[:edge_color] = log_size
    end

    if p.attributes.node_size[] == default_attrs.node_size[]
        kwargs[:node_size] = (log_flop/max_flop)*MAX_NODE_SIZE
    end

    if p.attributes.node_color[] == default_attrs.node_color[]
        kwargs[:node_color] = log_flop
    end

    f, ax, p = graphplot(graph;
    arrow_attr=(colorrange=(min_size, max_size), colormap=colormap),
    edge_attr=(colorrange=(min_size, max_size), colormap=colormap),
    node_attr=(colorrange=(min_flop, max_flop),
    colormap=to_colormap(:plasma)[begin:end-50]),
    kwargs...)

    # deactivate grid for 2D plots
    if !isa(ax, LScene)
        hidedecorations!(ax)
        hidespines!(ax)
        ax.aspect = DataAspect()
    end

    cbar = Colorbar(f[1,2], p.plots[1], label=L"\log_{2}(size)", flip_vertical_label=true, labelsize = 34)
    cbar.height = Relative(5/6)

    cbar2 = Colorbar(f[1,0], get_node_plot(p), label=L"\log_{10}(flops)", flipaxis=false, labelsize = 34)
    cbar2.height = Relative(5/6)

    cbar2.alignmode = Mixed(left = -10, right = -44)
    cbar.alignmode = Mixed(left = -44, right = -10)

    return f
end