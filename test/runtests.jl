using Test
import OptimizedEinsum

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(OptimizedEinsum, ambiguities=false, stale_deps=false)
    Aqua.test_ambiguities(OptimizedEinsum)
end