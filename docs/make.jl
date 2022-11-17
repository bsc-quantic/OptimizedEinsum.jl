using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

push!(LOAD_PATH, "$(@__DIR__)/..")

using Documenter
using OptimizedEinsum

DocMeta.setdocmeta!(OptimizedEinsum, :DocTestSetup, :(using OptimizedEinsum); recursive=true)

makedocs(
    modules=[OptimizedEinsum],
    sitename="OptimizedEinsum.jl",
    pages=Any[
        "Home"=>"index.md",
        "Path Solvers"=>Any[
            "Optimal"=>"optimal.md",
            "Greedy"=>"greedy.md",
        ],
        "Visualization"=>"viz.md",
    ]
)