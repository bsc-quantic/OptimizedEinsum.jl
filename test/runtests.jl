using Test
import OptimizedEinsum

using Aqua
Aqua.test_all(OptimizedEinsum, ambiguities=false, stale_deps=false)
Aqua.test_ambiguities(OptimizedEinsum)
