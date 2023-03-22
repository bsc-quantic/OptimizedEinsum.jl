using Test
import OptimizedEinsum

@testset "Unit tests" begin
    include("Utils_test.jl")
    include("Counters_test.jl")
    include("ContractionPath_test.jl")
    include("Visualization_test.jl")
end

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(OptimizedEinsum, ambiguities=false, stale_deps=false)
    Aqua.test_ambiguities(OptimizedEinsum)
end