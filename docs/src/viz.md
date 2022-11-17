# Visualization

We use `Graphs.jl` and `GraphMakie.jl` for visualization.

## Interactivity

```julia
# renders in a OpenGL context
using GLMakie

# renders in a browser
using WGLMakie
```

### 3D visualization

```julia
using NetworkLayout

draw(path, layout=Spring(dim=3))
```