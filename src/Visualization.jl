using Graphs
using Makie
using GraphMakie

const MIN_LOGFLOP = 0.01
const MAX_EDGE_WIDTH = 10.
const MAX_ARROW_SIZE = 25.
const MAX_NODE_SIZE = 40.

function Makie.plot(path::ContractionPath; colormap = to_colormap(:viridis)[begin:end-10], kwargs...)
    f = Figure()

    haskey(kwargs, :layout) && isa(kwargs[:layout],Spring{3}) ? ax = LScene(f[1,1]) : ax = Axis(f[1,1])

    scene = Makie.parent_scene(f)
    default_attrs = default_theme(scene, GraphPlot)

    kwargs = Dict{Symbol,Any}(kwargs)

    graph = SimpleDiGraph([Edge(from, to) for (pair, to) in zip(path.ssa_path, Iterators.countfrom(length(path.inputs) + 1)) for from in pair])

    log_size = [log2(size(path, i)) for i in 1:ne(graph)]
    log_flop = [(x = log10(flops(path, i)); isinf(x) ? MIN_LOGFLOP : x) for i in 1:nv(graph)]

    min_size, max_size = extrema(log_size)
    min_flop, max_flop = extrema(log_flop)

    kwargs[:edge_width] = (log_size/max_size)*MAX_EDGE_WIDTH
    kwargs[:arrow_size] = (log_size/max_size)*MAX_ARROW_SIZE
    kwargs[:edge_color] = log_size
    kwargs[:node_size] = (log_flop/max_flop)*MAX_NODE_SIZE
    kwargs[:node_color] = log_flop

    p  =  graphplot!(f[1,1], graph;
    arrow_attr = (colorrange=(min_size, max_size), colormap=colormap),
    edge_attr = (colorrange=(min_size, max_size), colormap=colormap),
    node_attr = (colorrange=(min_flop, max_flop),
    colormap = to_colormap(:plasma)[begin:end-50]), kwargs...)

    if !isa(ax, LScene) # hide decorations if it is not a 3D plot
        hidedecorations!(ax)
        hidespines!(ax)
        ax.aspect = DataAspect()
    end

    cbar = Colorbar(f[1,2], get_edge_plot(p), label=L"\log_{2}(size)", flip_vertical_label=true, labelsize = 34)
    cbar.height = Relative(5/6)

    cbar2 = Colorbar(f[1,0], get_node_plot(p), label=L"\log_{10}(flops)", flipaxis=false, labelsize = 34)
    cbar2.height = Relative(5/6)

    cbar2.alignmode = Mixed(left = -10, right = -30)
    cbar.alignmode = Mixed(left = -30, right = -10)

    display(f)

    return f, ax, p
end

# TODO replace `to_colormap(:viridis)[begin:end-10]` with a custom colormap
function Makie.plot!(f::GridPosition, path::ContractionPath; colormap = to_colormap(:viridis)[begin:end-10], kwargs...)
    scene = Scene()
    default_attrs = default_theme(scene, GraphPlot)

    kwargs = Dict{Symbol,Any}(kwargs)

    graph = SimpleDiGraph([Edge(from, to) for (pair, to) in zip(path.ssa_path, Iterators.countfrom(length(path.inputs) + 1)) for from in pair])

    log_size = [log2(size(path, i)) for i in 1:ne(graph)]
    log_flop = [(x = log10(flops(path, i)); isinf(x) ? MIN_LOGFLOP : x) for i in 1:nv(graph)]

    min_size, max_size = extrema(log_size)
    min_flop, max_flop = extrema(log_flop)

    kwargs[:edge_width] = (log_size/max_size)*MAX_EDGE_WIDTH
    kwargs[:arrow_size] = (log_size/max_size)*MAX_ARROW_SIZE
    kwargs[:edge_color] = log_size
    kwargs[:node_size] = (log_flop/max_flop)*MAX_NODE_SIZE
    kwargs[:node_color] = log_flop

    haskey(kwargs, :layout) && isa(kwargs[:layout],Spring{3}) ? ax = LScene(f[1,1]) : ax = Axis(f[1,1])

    p = graphplot!(f[1,1], graph;
    arrow_attr = (colorrange=(min_size, max_size), colormap=colormap),
    edge_attr = (colorrange=(min_size, max_size), colormap=colormap),
    node_attr = (colorrange=(min_flop, max_flop),
    # TODO replace `to_colormap(:plasma)[begin:end-50]), kwargs...)` with a custom colormap
    colormap = to_colormap(:plasma)[begin:end-50]), kwargs...)

    ax = current_axis(current_figure())

    # hide decorations if it is not a 3D plot
    if !isa(ax, LScene)
        hidedecorations!(ax)
        hidespines!(ax)
        ax.aspect = DataAspect()
    end

    # TODO configurable `labelsize`
    cbar = Colorbar(f[1,2], get_edge_plot(p), label="\\log\_\{2\}(size)", flip_vertical_label=true, labelsize = 34)
    cbar.height = Relative(5/6)

    cbar2 = Colorbar(f[1,0], get_node_plot(p), label="\\log\_\{10\}(flops)", flipaxis=false, labelsize = 34)
    cbar2.height = Relative(5/6)

    # TODO configurable alignments
    cbar2.alignmode = Mixed(left = -10, right = -30)
    cbar.alignmode = Mixed(left = -30, right = -10)

    return f
end